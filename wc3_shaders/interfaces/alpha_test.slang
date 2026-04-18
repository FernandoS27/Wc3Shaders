// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Fernando Sahmkow
// See LICENSE in the repository root for full terms.
//
// Compile-time alpha-test policy. The engine picks `AlphaTestOn` for
// cut-out materials (leaves, grates, decals) and `AlphaTestOff`
// elsewhere; specialization removes the branch and discard entirely for
// opaque and straight-alpha-blended draws.

implementing wc3_shaders;

public interface IAlphaTest
{
    // True when the policy emits a `discard`; consumers gate extra work
    // (e.g. the depth prepass computing a real alpha) on this flag.
    static const bool enabled;
    static void test(float alpha, float alphaRef);
};

// No-op: opaque / regular alpha-blended draws.
public struct AlphaTestOff : IAlphaTest
{
    static const bool enabled = false;
    static void test(float alpha, float alphaRef) { }
};

// Cut-out alpha test: `discard` if the computed alpha is below the
// reference threshold. The subtraction-based form (rather than `<`)
// matches the codegen of the original engine shaders.
public struct AlphaTestOn : IAlphaTest
{
    static const bool enabled = true;
    static void test(float alpha, float alphaRef)
    {
        if (alpha - alphaRef < 0)
            discard;
    }
};
