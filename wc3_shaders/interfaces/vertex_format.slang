// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Compile-time description of which optional attributes a given vertex
// format actually carries. Even though `VSInput` declares every possible
// attribute (so the input-element layout can stay fixed), the vertex
// shaders gate tangent transforms, UV transforms, and vertex-color
// modulation on these flags so dead code is eliminated per permutation.

implementing wc3_shaders;

public interface IVertexFormat
{
    static const bool hasTangent;
    static const bool hasVertexColor;
    static const bool hasUV0;
    static const bool hasUV1;
};

// Concrete format generator. The four bool generics make a 16-way
// specialization space; the engine only actually instantiates the
// combinations it needs.
public struct VertexFormat<
    let HAS_TANGENT : bool,
    let HAS_COLOR   : bool,
    let HAS_UV0     : bool,
    let HAS_UV1     : bool
> : IVertexFormat
{
    static const bool hasTangent    = HAS_TANGENT;
    static const bool hasVertexColor = HAS_COLOR;
    static const bool hasUV0        = HAS_UV0;
    static const bool hasUV1        = HAS_UV1;
};
