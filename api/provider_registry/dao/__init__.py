"""Provider registry DAO layer.

Async SQLite-backed data access for the provider registry subsystem.
All DAOs operate on aiosqlite connections and return plain dicts.
"""
from api.provider_registry.dao.db import get_connection, initialize_database
from api.provider_registry.dao.schema import SCHEMA_VERSION
from api.provider_registry.dao.provider_instances_dao import ProviderInstancesDAO
from api.provider_registry.dao.provider_credentials_dao import ProviderCredentialsDAO
from api.provider_registry.dao.provider_models_cache_dao import ProviderModelsCacheDAO
from api.provider_registry.dao.provider_usage_cache_dao import ProviderUsageCacheDAO
from api.provider_registry.dao.provider_sync_state_dao import ProviderSyncStateDAO
from api.provider_registry.dao.schema_migrations_dao import SchemaMigrationsDAO

__all__ = [
    "get_connection",
    "initialize_database",
    "SCHEMA_VERSION",
    "ProviderInstancesDAO",
    "ProviderCredentialsDAO",
    "ProviderModelsCacheDAO",
    "ProviderUsageCacheDAO",
    "ProviderSyncStateDAO",
    "SchemaMigrationsDAO",
]
