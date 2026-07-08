#!/usr/bin/env python

# Copyright 2025 The XenseRobotics Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import flexiv_rt

    from .config_bi_flexiv_rizon4_rt import BiFlexivRizon4RTConfig  # noqa: F401

    # Export flexiv_rt types for direct access
    Mode = flexiv_rt.Mode  # noqa: F401
    CoordType = flexiv_rt.CoordType  # noqa: F401
except ImportError:
    # Keep unrelated robot types usable when the optional Flexiv SDK is absent.
    pass


def __getattr__(name: str):
    """Load the hardware implementation only when it is requested."""
    if name == "BiFlexivRizon4RT":
        from .bi_flexiv_rizon4_rt import BiFlexivRizon4RT

        return BiFlexivRizon4RT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
