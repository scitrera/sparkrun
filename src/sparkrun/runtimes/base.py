"""Base class for sparkrun runtimes."""

from __future__ import annotations

from abc import abstractmethod
from logging import Logger
from typing import Any, TYPE_CHECKING

from scitrera_app_framework.api.plugins import Plugin
from scitrera_app_framework.api.variables import Variables

if TYPE_CHECKING:
    from sparkrun.recipe import Recipe

EXT_RUNTIME = "sparkrun.runtime"


class RuntimePlugin(Plugin):
    """Abstract base class for sparkrun inference runtimes.

    Each runtime is an SAF Plugin that registers as a multi-extension
    under the 'sparkrun.runtime' extension point. Multiple runtimes
    can coexist simultaneously.

    Subclasses must define:
        - runtime_name: str identifier (e.g. "vllm", "sglang")
        - generate_command(): produce the serve command from a recipe
        - resolve_container(): determine which container image to use
    """

    eager = False  # don't initialize until requested

    # --- Subclass must define ---
    runtime_name: str = ""
    default_image_prefix: str = ""

    # --- SAF Plugin interface ---

    def name(self) -> str:
        return "sparkrun.runtime.%s" % self.runtime_name

    def extension_point_name(self, v: Variables) -> str:
        return EXT_RUNTIME

    def is_enabled(self, v: Variables) -> bool:
        # Must return False for multi-extension plugins to prevent SAF's
        # single-extension cache (er[ext_name]) from short-circuiting
        # subsequent plugin initializations under the same extension point.
        return False

    def is_multi_extension(self, v: Variables) -> bool:
        return True

    def initialize(self, v: Variables, logger: Logger) -> RuntimePlugin:
        return self

    # --- Runtime interface ---

    @abstractmethod
    def generate_command(self, recipe: Recipe, overrides: dict[str, Any],
                         is_cluster: bool, num_nodes: int = 1,
                         head_ip: str | None = None) -> str:
        """Generate the serve command string from recipe + CLI overrides.

        Args:
            recipe: The loaded recipe
            overrides: CLI override values (e.g. --port 9000)
            is_cluster: Whether running in multi-node mode
            num_nodes: Total number of nodes in the cluster
            head_ip: Head node IP (only set for cluster mode)

        Returns:
            The full command string to execute inside the container
        """
        ...

    @abstractmethod
    def resolve_container(self, recipe: Recipe, overrides: dict[str, Any] | None = None) -> str:
        """Resolve the container image to use.

        Args:
            recipe: The loaded recipe
            overrides: Optional CLI overrides

        Returns:
            Fully qualified container image reference
        """
        ...

    def get_cluster_env(self, head_ip: str, num_nodes: int) -> dict[str, str]:
        """Return runtime-specific environment variables for cluster mode.

        Override in subclasses to inject runtime-specific cluster config.
        """
        return {}

    def validate_recipe(self, recipe: Recipe) -> list[str]:
        """Return list of warnings/errors for runtime-specific fields.

        Override in subclasses to add runtime-specific validation.
        """
        return []

    def is_delegating_runtime(self) -> bool:
        """True if this runtime delegates entirely to external scripts.

        Delegating runtimes (like eugr-vllm) bypass sparkrun's orchestration
        layer and instead call external tools directly.
        """
        return False

    def __repr__(self) -> str:
        return "%s(runtime_name=%r)" % (self.__class__.__name__, self.runtime_name)
