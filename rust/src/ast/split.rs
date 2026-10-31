//! §14 fix: split `finalDensity` into its interpolated-boundary subtrees
//! (computed at cell corners) and the outer tree (evaluated per block).
//!
//! MC semantics: `outer(trilinear(inner))`, not `trilinear(outer(inner))`.
//! We extract each `InterpolatedNoiseSampler` (MC's `Wrapping.INTERPOLATED`)
//! subtree into `inners`, replacing it with `AstNode::InnerRef(i)`.

use crate::ast::AstNode;

/// Result of splitting one channel: the extracted inner subtrees (in
/// traversal order) and the outer tree with `InnerRef(i)` placeholders.
pub struct Split {
    pub inners: Vec<AstNode>,
    pub outer: AstNode,
}

pub fn split(node: &AstNode, inners: &mut Vec<AstNode>) -> AstNode {
    use AstNode::*;
    match node {
        // Leaf — nothing to split.
        Constant(_) | InputX | InputY | InputZ
        | SimplexNoise2D { .. } | SimplexNoise3D { .. } | InterpNoiseSampler { .. }
        | EndIslands { .. } | Beardifier { .. } | YClampedGradient { .. }
        | BiomeTree { .. } | NoiseSampler { .. } | InnerRef(_) => node.clone(),

        // `ExternalInput` is a corner-grid quantity, not a per-block one.
        //
        // The Java side pre-evaluates every external subtree on the same
        // `gx * gy * gz` lattice the interpolated noise uses
        // (`WgpuBatchCoordinator.uploadExternalInputs`), and the WGSL reads
        // those values back by cell index. Its semantics are therefore
        // identical to an `InterpolatedNoiseSampler`: sample at the corners,
        // then trilinearly interpolate.
        //
        // Leaving it in the outer tree broke that contract only on specific
        // worlds: the outer tree is evaluated per block, and
        // `external_inputs` is bound only in the corner shards. A tree with an
        // external node in its outer half (observed with a Twilight Forest
        // worldgen) made the BlockOut shader reference an undeclared
        // `external_inputs`, fail validation, and leave the block buffer
        // uninitialised.
        ExternalInput { .. } => {
            let idx = inners.len();
            inners.push(node.clone());
            InnerRef(idx)
        }

        // The boundary: extract the whole subtree, replace with a ref.
        // Both `InterpolatedNoiseSampler` and `CacheInterpolated` are MC's
        // corner-grid interpolators. Non-linear nodes wrapped *outside* them
        // must be evaluated per block on the interpolated result, so these
        // subtrees must be split out.
        InterpolatedNoiseSampler { child } => {
            let idx = inners.len();
            inners.push((**child).clone());
            InnerRef(idx)
        }
        CacheInterpolated(child) => {
            let idx = inners.len();
            inners.push((**child).clone());
            InnerRef(idx)
        }

        // Recurse.
        Add(a, b) => Add(Box::new(split(a, inners)), Box::new(split(b, inners))),
        Sub(a, b) => Sub(Box::new(split(a, inners)), Box::new(split(b, inners))),
        Mul(a, b) => Mul(Box::new(split(a, inners)), Box::new(split(b, inners))),
        Div(a, b) => Div(Box::new(split(a, inners)), Box::new(split(b, inners))),
        Min(a, b) => Min(Box::new(split(a, inners)), Box::new(split(b, inners))),
        Max(a, b) => Max(Box::new(split(a, inners)), Box::new(split(b, inners))),
        FloorDiv(a, b) => FloorDiv(Box::new(split(a, inners)), Box::new(split(b, inners))),
        Abs(a) => Abs(Box::new(split(a, inners))),
        Square(a) => Square(Box::new(split(a, inners))),
        Cube(a) => Cube(Box::new(split(a, inners))),
        Sqrt(a) => Sqrt(Box::new(split(a, inners))),
        Squeeze(a) => Squeeze(Box::new(split(a, inners))),
        CacheFlat(a) => CacheFlat(Box::new(split(a, inners))),
        Cache2D(a) => Cache2D(Box::new(split(a, inners))),
        Shift { kind, offset } => Shift { kind: *kind, offset: Box::new(split(offset, inners)) },
        WeirdScaledSampler { mapper, input, noise } => WeirdScaledSampler {
            mapper: *mapper, input: Box::new(split(input, inners)),
            noise: Box::new(split(noise, inners)) },
        ShiftedNoise { xz_scale, y_scale, shift_x, shift_y, shift_z, noise } => ShiftedNoise {
            xz_scale: *xz_scale, y_scale: *y_scale,
            shift_x: Box::new(split(shift_x, inners)),
            shift_y: Box::new(split(shift_y, inners)),
            shift_z: Box::new(split(shift_z, inners)),
            noise: Box::new(split(noise, inners)) },
        Spline { locations, derivatives, location, values } => Spline {
            locations: locations.clone(), derivatives: derivatives.clone(),
            location: Box::new(split(location, inners)),
            values: values.iter().map(|v| split(v, inners)).collect() },
        RangeChoice { input, min, max, in_bounds, out_of_bounds_lo, out_of_bounds_hi } =>
            RangeChoice {
                input: Box::new(split(input, inners)),
                min: Box::new(split(min, inners)),
                max: Box::new(split(max, inners)),
                in_bounds: Box::new(split(in_bounds, inners)),
                out_of_bounds_lo: Box::new(split(out_of_bounds_lo, inners)),
                out_of_bounds_hi: Box::new(split(out_of_bounds_hi, inners)) },
        IntervalSelect { input, min, max, in_bounds, out_of_bounds_lo, out_of_bounds_hi } =>
            IntervalSelect {
                input: Box::new(split(input, inners)),
                min: Box::new(split(min, inners)),
                max: Box::new(split(max, inners)),
                in_bounds: Box::new(split(in_bounds, inners)),
                out_of_bounds_lo: Box::new(split(out_of_bounds_lo, inners)),
                out_of_bounds_hi: Box::new(split(out_of_bounds_hi, inners)) },
        FindTopSurface { density, initial, min_y, max_y } => FindTopSurface {
            density: Box::new(split(density, inners)),
            initial: Box::new(split(initial, inners)),
            min_y: *min_y, max_y: *max_y },
        Aquifer { density, fluid_level } => Aquifer {
            density: Box::new(split(density, inners)),
            fluid_level: Box::new(split(fluid_level, inners)) },
        Root(chs) => Root(chs.iter().map(|c| split(c, inners)).collect()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::AstNode;

    /// `ExternalInput` must be treated as a corner-grid boundary, like
    /// `InterpolatedNoiseSampler`, not as a leaf.
    ///
    /// The Java side pre-evaluates external subtrees on the `gx * gy * gz`
    /// lattice and the WGSL reads them back by cell index, so the values only
    /// exist at corners. Leaving the node in the outer tree made the BlockOut
    /// pass — which has no `external_inputs` binding — emit a read of it,
    /// failing shader validation and leaving the block buffer uninitialised.
    #[test]
    fn external_input_is_extracted_as_a_boundary() {
        let mut inners = Vec::new();
        let node = AstNode::ExternalInput { index: 3 };

        let outer = split(&node, &mut inners);

        assert_eq!(inners.len(), 1, "the external subtree must be extracted");
        assert_eq!(inners[0], node, "the extracted node must be the input itself");
        assert_eq!(outer, AstNode::InnerRef(0), "the outer tree must see a ref");
    }

    /// The same node nested inside arithmetic must still be pulled out, and
    /// the surrounding structure preserved.
    #[test]
    fn external_input_nested_in_arithmetic_is_extracted() {
        let mut inners = Vec::new();
        let ext = AstNode::ExternalInput { index: 0 };
        let node = AstNode::Add(
            Box::new(AstNode::Constant(1.0)),
            Box::new(ext.clone()),
        );

        let outer = split(&node, &mut inners);

        assert_eq!(inners.len(), 1);
        assert_eq!(inners[0], ext);
        assert_eq!(
            outer,
            AstNode::Add(
                Box::new(AstNode::Constant(1.0)),
                Box::new(AstNode::InnerRef(0)),
            )
        );
    }

    /// Guard against the regression this test was written for: a genuine leaf
    /// must still be left in place, not extracted.
    #[test]
    fn plain_leaves_are_not_extracted() {
        let mut inners = Vec::new();
        let outer = split(&AstNode::InputX, &mut inners);
        assert!(inners.is_empty(), "InputX is not a boundary");
        assert_eq!(outer, AstNode::InputX);
    }

    /// Two external nodes must get distinct slots, in traversal order.
    #[test]
    fn multiple_external_inputs_get_distinct_indices() {
        let mut inners = Vec::new();
        let a = AstNode::ExternalInput { index: 0 };
        let b = AstNode::ExternalInput { index: 1 };
        let node = AstNode::Add(Box::new(a.clone()), Box::new(b.clone()));

        let outer = split(&node, &mut inners);

        assert_eq!(inners.len(), 2);
        assert_eq!(inners[0], a);
        assert_eq!(inners[1], b);
        assert_eq!(
            outer,
            AstNode::Add(
                Box::new(AstNode::InnerRef(0)),
                Box::new(AstNode::InnerRef(1)),
            )
        );
    }
}
