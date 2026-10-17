/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 RubidiumLighting
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

//! MC `Spline` emitter — cubic Hermite interpolation over a recursive tree.
//!
//! Confirmed from bytecode (`Spline$Implementation.apply`):
//! ```text
//! t = locationFunction.apply(pos)
//! k = (first i where t < locations[i]) - 1
//! if k < 0:      return outside(t, values[0], derivatives[0], locations[0])
//! if k == n-1:   return outside(t, values[n-1], derivatives[n-1], locations[n-1])
//! p  = (t - locations[k]) / (locations[k+1] - locations[k])
//! a  =  derivatives[k]  *(l[k+1]-l[k]) - (v[k+1]-v[k])
//! b  = -derivatives[k+1]*(l[k+1]-l[k]) + (v[k+1]-v[k])
//! return lerp(p, v[k], v[k+1]) + p*(1-p)*lerp(p, a, b)
//!
//! outside(t, v, d, l) = (d == 0) ? v : v + d*(t - l)
//! ```
//! (lerp(t,a,b) = a + t*(b-a), no clamp.)

use crate::ast::AstNode;
use crate::codegen::wgsl_gen::GenCtx;
use crate::Result;
use anyhow::anyhow;

pub fn emit(node: &AstNode, ctx: &mut GenCtx) -> Result<String> {
    let AstNode::Spline { locations, derivatives, location, values } = node else {
        return Err(anyhow!("spline emitter called on non-spline node"));
    };
    let n = locations.len();
    if n == 0 || derivatives.len() != n || values.len() != n {
        return Err(anyhow!("Spline shape mismatch: loc={n} der={} val={}", derivatives.len(), values.len()));
    }

    // 1) t = location function value
    let t = crate::codegen::emitters::emit_node(location, ctx)?;

    // 2) evaluate every per-node value (post-order recursion)
    let mut vs: Vec<String> = Vec::with_capacity(n);
    for v in values {
        vs.push(crate::codegen::emitters::emit_node(v, ctx)?);
    }

    let result = ctx.fresh_var("spline");

    if n == 1 {
        // MC `Spline$Implementation.apply`: for n==1, k is -1 (t<l0) or
        // n-1==0 (t>=l0), both route to `outside()`. With derivative d0 that
        // extrapolates v0 + d0*(t-l0); with d0==0 it is the constant v0.
        let l0 = locations[0];
        let d0 = derivatives[0];
        if d0 == 0.0 {
            ctx.emit(format!("let {result}: f32 = {};", vs[0]));
        } else {
            // Reuse `t` already emitted above (line 57): result = v0 + d0*(t-l0).
            ctx.emit(format!(
                "let {result}: f32 = {vs0} + ({d0:.9e}) * ({t} - {l0:.9e});",
                vs0 = vs[0]
            ));
        }
        return Ok(result);
    }

    // outside-low (index 0)
    let l0 = locations[0];
    let d0 = derivatives[0];
    let low_expr = if d0 == 0.0 {
        vs[0].clone()
    } else {
        format!("{} + ({d0:.9e}) * ({t} - {l0:.9e})", vs[0])
    };
    // outside-high (index n-1)
    let ln = locations[n - 1];
    let dn = derivatives[n - 1];
    let high_expr = if dn == 0.0 {
        vs[n - 1].clone()
    } else {
        format!("{} + ({dn:.9e}) * ({t} - {ln:.9e})", vs[n - 1])
    };

    ctx.emit(format!("var {result}: f32 = 0.0;"));
    ctx.emit(format!("if ({t} < {l0:.9e}) {{"));
    ctx.emit(format!("  {result} = {low_expr};"));
    ctx.emit(format!("}} else if ({t} >= {ln:.9e}) {{"));
    ctx.emit(format!("  {result} = {high_expr};"));
    ctx.emit("} else {");

    for k in 0..(n - 1) {
        let lk = locations[k];
        let lk1 = locations[k + 1];
        let dk = derivatives[k];
        let dk1 = derivatives[k + 1];
        let vk = &vs[k];
        let vk1 = &vs[k + 1];
        let width = lk1 - lk;

        if k == 0 {
            ctx.emit(format!("  if ({t} < {lk1:.9e}) {{"));
        } else if k == n - 2 {
            ctx.emit("  else {".to_string());
        } else {
            ctx.emit(format!("  else if ({t} < {lk1:.9e}) {{"));
        }
        // One call instead of four inline lines; spline_seg is bit-identical
        // to the previous expansion (see math_lib.wgsl).
        ctx.emit(format!(
            "    {result} = spline_seg({t}, {lk:.9e}, {width:.9e}, {dk:.9e}, {dk1:.9e}, {vk}, {vk1});"
        ));
        ctx.emit("  }".to_string());
    }
    ctx.emit("}".to_string());

    Ok(result)
}
