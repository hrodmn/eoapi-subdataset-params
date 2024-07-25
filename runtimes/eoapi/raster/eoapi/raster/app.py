"""eoAPI Raster application."""

import logging
import re
import urllib.parse
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

import attr
import jinja2
import pystac
from eoapi.raster import __version__ as eoapi_raster_version
from eoapi.raster.config import ApiSettings
from fastapi import Depends, FastAPI, Path, Query
from psycopg import OperationalError
from psycopg.rows import dict_row
from psycopg_pool import PoolTimeout
from rio_tiler.io import BaseReader, Reader
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from starlette_cramjam.middleware import CompressionMiddleware
from titiler.core.dependencies import DefaultDependency
from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers
from titiler.core.factory import (
    AlgorithmFactory,
    ColorMapFactory,
    MultiBaseTilerFactory,
    TilerFactory,
    TMSFactory,
)
from titiler.core.middleware import CacheControlMiddleware
from titiler.extensions.viewer import cogViewerExtension
from titiler.mosaic.errors import MOSAIC_STATUS_CODES
from titiler.pgstac import mosaic, reader
from titiler.pgstac.db import close_db_connection, connect_to_db
from titiler.pgstac.dependencies import (
    CollectionIdParams,
    ItemIdParams,
    SearchIdParams,
    get_stac_item,
)
from titiler.pgstac.extensions import searchInfoExtension
from titiler.pgstac.factory import (
    MosaicTilerFactory,
    add_search_list_route,
    add_search_register_route,
)
from typing_extensions import Annotated

logging.getLogger("botocore.credentials").disabled = True
logging.getLogger("botocore.utils").disabled = True
logging.getLogger("rio-tiler").setLevel(logging.ERROR)

settings = ApiSettings()

jinja2_env = jinja2.Environment(
    loader=jinja2.ChoiceLoader(
        [
            jinja2.PackageLoader(__package__, "templates"),
        ]
    )
)
templates = Jinja2Templates(env=jinja2_env)


@dataclass(init=False)
class ReaderParams(DefaultDependency):
    """reader parameters."""

    reader_options: Dict = field(init=False)

    def __init__(
        self,
        subdataset_name: Annotated[
            Optional[str],
            Query(
                title="Subdataset name",
                description="The name of a subdataset within the asset.",
            ),
        ] = None,
        subdataset_bands: Annotated[
            Optional[List[int]],
            Query(
                title="Subdataset bands",
                description="The band index of a subdataset within the asset.",
            ),
        ] = None,
    ):
        """Initialize ReaderParams"""
        params = {}
        if subdataset_name:
            params["subdataset_name"] = subdataset_name

        if subdataset_bands:
            params["subdataset_bands"] = subdataset_bands  # type: ignore

        self.reader_options = params


@attr.s
class CustomReader(Reader):
    subdataset_name: Optional[str] = attr.ib(default=None)
    subdataset_bands: Optional[List[int]] = attr.ib(default=None)

    def __attrs_post_init__(self):
        vrt_params = {}
        if self.subdataset_name:
            vrt_params["sd_name"] = self.subdataset_name

        if self.subdataset_bands:
            vrt_params["bands"] = ",".join(
                [str(band) for band in self.subdataset_bands]
            )

        if vrt_params:
            params = urllib.parse.urlencode(vrt_params, safe=",")
            self.input = f"vrt:///vsicurl/{self.input}?{params}"
        super().__attrs_post_init__()


@attr.s
class PgSTACReader(reader.PgSTACReader):
    reader: Type[BaseReader] = attr.ib(default=CustomReader)


@attr.s
class CustomSTACReader(mosaic.CustomSTACReader):
    reader: Type[BaseReader] = attr.ib(default=CustomReader)


@attr.s
class PGSTACBackend(mosaic.PGSTACBackend):
    reader: Type[BaseReader] = attr.ib(init=False, default=CustomSTACReader)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan."""
    # Create Connection Pool
    await connect_to_db(app)
    yield
    # Close the Connection Pool
    await close_db_connection(app)


app = FastAPI(
    title=settings.name,
    version=eoapi_raster_version,
    openapi_url="/api",
    docs_url="/api.html",
    root_path=settings.root_path,
    lifespan=lifespan,
)
add_exception_handlers(app, DEFAULT_STATUS_CODES)
add_exception_handlers(app, MOSAIC_STATUS_CODES)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=["*"],
    )

app.add_middleware(
    CacheControlMiddleware,
    cachecontrol=settings.cachecontrol,
    exclude_path={r"/healthz", r"/collections"},
)
app.add_middleware(
    CompressionMiddleware,
    exclude_mediatype={
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/jp2",
        "image/webp",
    },
)


###############################################################################
# `Secret` endpoint for mosaic builder. Do not need to be public (in the OpenAPI docs)
@app.get("/collections", include_in_schema=False)
async def list_collection(request: Request):
    """list collections."""
    with request.app.state.dbpool.connection() as conn:
        with conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute("SELECT * FROM pgstac.all_collections();")
            r = cursor.fetchone()
            return r.get("all_collections", [])


###############################################################################
# STAC Search Endpoints
searches = MosaicTilerFactory(
    reader=PGSTACBackend,
    reader_dependency=ReaderParams,
    path_dependency=SearchIdParams,
    router_prefix="/searches/{search_id}",
    add_statistics=True,
    add_viewer=True,
    add_part=True,
    extensions=[
        searchInfoExtension(),
    ],
)
app.include_router(
    searches.router, tags=["STAC Search"], prefix="/searches/{search_id}"
)

add_search_register_route(
    app,
    prefix="/searches",
    tile_dependencies=[
        searches.layer_dependency,
        searches.dataset_dependency,
        searches.pixel_selection_dependency,
        searches.tile_dependency,
        searches.process_dependency,
        searches.rescale_dependency,
        searches.colormap_dependency,
        searches.render_dependency,
        searches.pgstac_dependency,
        searches.reader_dependency,
        searches.backend_dependency,
    ],
    tags=["STAC Search"],
)
add_search_list_route(app, prefix="/searches", tags=["STAC Search"])


@app.get("/searches/builder", response_class=HTMLResponse, tags=["STAC Search"])
async def virtual_mosaic_builder(request: Request):
    """Mosaic Builder Viewer."""
    base_url = str(request.base_url)
    return templates.TemplateResponse(
        name="mosaic-builder.html",
        context={
            "request": request,
            "register_endpoint": str(
                app.url_path_for("register_search").make_absolute_url(base_url=base_url)
            ),
            "collections_endpoint": str(
                app.url_path_for("list_collection").make_absolute_url(base_url=base_url)
            ),
        },
        media_type="text/html",
    )


###############################################################################
# STAC COLLECTION Endpoints
collection = MosaicTilerFactory(
    reader=PGSTACBackend,
    reader_dependency=ReaderParams,
    path_dependency=CollectionIdParams,
    router_prefix="/collections/{collection_id}",
    add_statistics=True,
    add_viewer=True,
    add_part=True,
    extensions=[
        searchInfoExtension(),
    ],
)
app.include_router(
    collection.router, tags=["STAC Collection"], prefix="/collections/{collection_id}"
)


###############################################################################
# STAC Item Endpoints
stac = MultiBaseTilerFactory(
    reader=PgSTACReader,
    reader_dependency=ReaderParams,
    path_dependency=ItemIdParams,
    router_prefix="/collections/{collection_id}/items/{item_id}",
    add_viewer=True,
)
app.include_router(
    stac.router,
    tags=["STAC Item"],
    prefix="/collections/{collection_id}/items/{item_id}",
)


@stac.router.get("/viewer", response_class=HTMLResponse)
def viewer(request: Request, item: pystac.Item = Depends(stac.path_dependency)):
    """STAC Viewer

    Simplified version of https://github.com/developmentseed/titiler/blob/main/src/titiler/extensions/titiler/extensions/templates/stac_viewer.html
    """
    return templates.TemplateResponse(
        name="stac-viewer.html",
        context={
            "request": request,
            "endpoint": request.url.path.replace("/viewer", ""),
        },
        media_type="text/html",
    )


app.include_router(
    stac.router,
    tags=["STAC Item"],
    prefix="/collections/{collection_id}/items/{item_id}",
)


def ItemAssetIdParams(
    request: Request,
    collection_id: Annotated[
        str,
        Path(description="STAC Collection Identifier"),
    ],
    item_id: Annotated[str, Path(description="STAC Item Identifier")],
    asset_id: Annotated[str, Path(description="STAC Item Identifier")],
    subdataset_name: Annotated[
        Optional[str],
        Query(
            title="Subdataset name",
            description="The name of a subdataset within the asset.",
        ),
    ] = None,
    subdataset_bands: Annotated[
        Optional[List[int]],
        Query(
            title="Subdataset bands",
            description="The band index of a subdataset within the asset.",
        ),
    ] = None,
):
    """STAC Item Asset dependency."""
    item = get_stac_item(request.app.state.dbpool, collection_id, item_id)
    asset_info = item.assets[asset_id]
    url = asset_info.get_absolute_href() or asset_info.href

    vrt_params = {}
    if subdataset_name:
        vrt_params["sd_name"] = subdataset_name

    if subdataset_bands:
        vrt_params["bands"] = ",".join([str(band) for band in subdataset_bands])

    if vrt_params:
        params = urllib.parse.urlencode(vrt_params, safe=",")
        url = f"vrt:///vsicurl/{url}?{params}"

    return url


###############################################################################
# STAC Assets Endpoints
assets = TilerFactory(
    reader=Reader,
    path_dependency=ItemAssetIdParams,
    router_prefix="/collections/{collection_id}/items/{item_id}/assets/{asset_id}",
)
app.include_router(
    assets.router,
    tags=["STAC Asset"],
    prefix="/collections/{collection_id}/items/{item_id}/assets/{asset_id}",
)


###############################################################################
# External Dataset Endpoints
def SdsParams(
    url: Annotated[
        str,
        Query(
            ...,
            description="Dataset url",
        ),
    ],
    subdataset_name: Annotated[
        Optional[str],
        Query(
            title="Subdataset name",
            description="The name of a subdataset within the asset.",
        ),
    ] = None,
    subdataset_bands: Annotated[
        Optional[List[int]],
        Query(
            title="Subdataset bands",
            description="The band index of a subdataset within the asset.",
        ),
    ] = None,
):
    vrt_params = {}
    if subdataset_name:
        vrt_params["sd_name"] = subdataset_name

    if subdataset_bands:
        vrt_params["bands"] = ",".join([str(band) for band in subdataset_bands])

    if vrt_params:
        params = urllib.parse.urlencode(vrt_params, safe=",")
        url = f"vrt:///vsicurl/{url}?{params}"

    return url


assets = TilerFactory(
    reader=Reader,
    path_dependency=SdsParams,
    router_prefix="/dataset",
)

app.include_router(
    assets.router,
    tags=["External Dataset"],
    prefix="/dataset",
)

###############################################################################
# COG Endpoints
cog = TilerFactory(
    router_prefix="/cog",
    extensions=[
        cogViewerExtension(),
    ],
)

app.include_router(cog.router, prefix="/cog", tags=["Cloud Optimized GeoTIFF"])

###############################################################################
# Tiling Schemes Endpoints
tms = TMSFactory()
app.include_router(tms.router, tags=["Tiling Schemes"])

###############################################################################
# Algorithms Endpoints
algorithms = AlgorithmFactory()
app.include_router(algorithms.router, tags=["Algorithms"])

###############################################################################
# Colormaps endpoints
cmaps = ColorMapFactory()
app.include_router(
    cmaps.router,
    tags=["ColorMaps"],
)


###############################################################################
# Health Check Endpoint
@app.get("/healthz", description="Health Check", tags=["Health Check"])
def ping(
    timeout: int = Query(
        1, description="Timeout getting SQL connection from the pool."
    ),
) -> Dict:
    """Health check."""
    try:
        with app.state.dbpool.connection(timeout) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT version from pgstac.migrations;")
                version = cursor.fetchone()
        return {"database_online": True, "pgstac_version": version}
    except (OperationalError, PoolTimeout):
        return {"database_online": False}


###############################################################################
# Landing page Endpoint
@app.get(
    "/",
    response_class=HTMLResponse,
    tags=["Landing"],
)
def landing(request: Request):
    """Get landing page."""
    data = {
        "title": settings.name or "eoAPI-raster",
        "links": [
            {
                "title": "Landing page",
                "href": str(request.url_for("landing")),
                "type": "text/html",
                "rel": "self",
            },
            {
                "title": "the API definition (JSON)",
                "href": str(request.url_for("openapi")),
                "type": "application/vnd.oai.openapi+json;version=3.0",
                "rel": "service-desc",
            },
            {
                "title": "the API documentation",
                "href": str(request.url_for("swagger_ui_html")),
                "type": "text/html",
                "rel": "service-doc",
            },
            {
                "title": "eoAPI Virtual Mosaic list (JSON)",
                "href": str(app.url_path_for("list_searches")),
                "type": "application/json",
                "rel": "data",
            },
            {
                "title": "eoAPI Virtual Mosaic builder",
                "href": str(app.url_path_for("virtual_mosaic_builder")),
                "type": "text/html",
                "rel": "data",
            },
            {
                "title": "eoAPI Virtual Mosaic viewer (template URL)",
                "href": str(app.url_path_for("map_viewer", search_id="{search_id}")),
                "type": "text/html",
                "rel": "data",
                "templated": True,
            },
            {
                "title": "eoAPI Collection viewer (template URL)",
                "href": str(
                    app.url_path_for("map_viewer", collection_id="{collection_id}")
                ),
                "type": "text/html",
                "rel": "data",
                "templated": True,
            },
            {
                "title": "eoAPI Item viewer (template URL)",
                "href": str(
                    app.url_path_for(
                        "map_viewer",
                        collection_id="{collection_id}",
                        item_id="{item_id}",
                    )
                ),
                "type": "text/html",
                "rel": "data",
                "templated": True,
            },
        ],
    }

    urlpath = request.url.path
    if root_path := request.app.root_path:
        urlpath = re.sub(r"^" + root_path, "", urlpath)
    crumbs = []
    baseurl = str(request.base_url).rstrip("/")

    crumbpath = str(baseurl)
    for crumb in urlpath.split("/"):
        crumbpath = crumbpath.rstrip("/")
        part = crumb
        if part is None or part == "":
            part = "Home"
        crumbpath += f"/{crumb}"
        crumbs.append({"url": crumbpath.rstrip("/"), "part": part.capitalize()})

    return templates.TemplateResponse(
        request,
        name="landing.html",
        context={
            "request": request,
            "response": data,
            "template": {
                "api_root": baseurl,
                "params": request.query_params,
                "title": "TiTiler-PgSTAC",
            },
            "crumbs": crumbs,
            "url": str(request.url),
            "baseurl": baseurl,
            "urlpath": str(request.url.path),
            "urlparams": str(request.url.query),
        },
    )
