#!/usr/bin/env python3

# Import standard de python3
import numpy as np

from deMonPy.module import modules


class _relax_geometry(modules):
    def __init__(self, context, **kwargs):

        super().__init__(context=context, **kwargs)

        self._module_parameters = None

    def restart(self, **kwds):

        image = kwds.pop("image", None)
        if not image:
            image = self.context.results["output_geometry"]

        self._module_parameters.update(**kwds)

        self.forward(image=image, **self._module_parameters)

    def check_distances(self, threshold=0.7):
        """Return the minimum interatomic distance of the relaxed geometry.

        A distance below *threshold* (in angstrom) usually indicates a
        collapsed or unphysical structure; when that happens an error
        entry is recorded on the underlying output reader so that
        ``context.has_errors()`` reports it.

        Args:
            threshold: Minimum acceptable interatomic distance, in
                angstrom.  Defaults to ``0.7``.

        Returns:
            float | None: The minimum pairwise distance, or ``None`` when
            no output geometry is available yet.
        """
        image = self.context.results.get("output_geometry")
        if image is None:
            return None

        if hasattr(image, "get_positions"):
            positions = np.asarray(image.get_positions())
        else:
            positions = np.asarray(getattr(image, "positions", image))

        if positions.ndim != 2 or positions.shape[0] < 2:
            return None

        diff = positions[:, None, :] - positions[None, :, :]
        dmat = np.sqrt((diff**2).sum(axis=-1))
        iu = np.triu_indices(positions.shape[0], k=1)
        min_dist = float(dmat[iu].min())

        if min_dist < threshold:
            self.context._wo._add_error(
                "geometry",
                f"Minimum interatomic distance {min_dist:.3f} A is below "
                f"threshold {threshold} A (possible collapsed structure).",
            )
        return min_dist

    def is_converged(
        self,
    ):

        for line in self.context._wo.lines:
            if self.context._wo.is_inside("optimization not converged", line):
                return False

        return True

    def forward(self, image, max=999, algo="CGRAD", out=1, restart=False, **args):

        self._module_parameters = dict(
            max=max, algo=algo, out=out, restart=restart, **args
        )

        self.update_parameters(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {"OPT": {"MAX": max, algo: True, "OUT": out, **args}}
                }
            }
        )

        self.context.calculate(symbols=image.symbols, positions=image.positions)

        if not self.is_converged():
            self.context.results["converged"] = False
        else:
            self.context.results["converged"] = True
