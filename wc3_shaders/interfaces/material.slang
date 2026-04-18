// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Compile-time flag that switches between the plain PBR material and the
// team-color multi-layer variant. `hasMultiLayer` gates the extra
// team-color texture fetch + hue/luma blend in the pixel shader bodies,
// and selects the multi-layer variants of the cloak / fresnel overlays.

implementing wc3_shaders;

public interface IMaterialLayer
{
    static const bool hasMultiLayer;
};

// Standard single-layer PBR material. Used by terrain and most props.
public struct StandardMaterial : IMaterialLayer
{
    static const bool hasMultiLayer = false;
};

// Multi-layer material used by units and team-owned objects. A second
// "team color" albedo is fetched and blended in using the mask stored
// in ORM.w (see effects/multi_layer).
public struct MultiLayerMaterial : IMaterialLayer
{
    static const bool hasMultiLayer = true;
};
