#!/usr/bin/env python3

# Import standard de python3
from __future__ import annotations

from typing import Any, Sequence

from ase import Atoms

from deMonPy.module import modules


class _ptmc(modules):
    def __init__(self, context, **kwargs):

        super().__init__(context=context, **kwargs)

        self._module_parameters = None

    def ptmc(self, **kwds):

        return self.forward(**kwds)

    def mc(self, **kwds):
        kwds["min_temp"] = None
        kwds["max_temp"] = None
        kwds["n_temp"] = 1
        kwds["distribution_temp"] = ""
        return self.forward(**kwds)

    def forward(
        self,
        image: Atoms,
        max: int = 30,
        seed: bool = True,
        out: int = 1,
        wall: float = 8.0,
        temperature: float | None = None,
        n_temp: int = 10,
        max_temp: float = 300,
        min_temp: float = 30,
        rescale: float = 10,
        distribution_temp: str = "geom",
        temp_list: Sequence[float] | None = None,
        restart: bool = False,
        **args: Any,
    ) -> None:

        self._module_parameters = dict(
            max=max,
            seed=seed,
            out=out,
            restart=restart,
            temperature=temperature,
            n_temp=n_temp,
            max_temp=max_temp,
            min_temp=min_temp,
            distribution_temp=distribution_temp,
            rescale=rescale,
            temp_list=temp_list,
            **args,
        )

        if temp_list is not None:
            raise NotImplementedError("temp_list (LIST) is not implemented")

        if distribution_temp.lower() not in ["geom", "linear", ""]:
            raise ValueError(
                "Unknown temperature distribution "
                f"{distribution_temp!r}: expected 'geom' or 'linear'"
            )

        self.update_parameters(
            {
                "DEMON_MODULE": {
                    "ACTIVE": {
                        "PTMC": {
                            "MC": {"MAX": max, "SEED": seed, "WALL": wall},
                            "MCTEMP": {
                                "TMC": temperature,
                                "NTEMP": n_temp,
                                distribution_temp.upper(): True,
                                "TEMPMIN": min_temp,
                                "TEMPMAX": max_temp,
                                "RESCALE": rescale,
                                "SDBG": False,
                                "OUT": 1,
                                "SOUT": 1,
                            },
                        }
                    }
                }
            }
        )

        self.context.calculate(symbols=image.symbols, positions=image.positions)
