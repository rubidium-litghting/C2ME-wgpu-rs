/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2021-2026 RubidiumLighting
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

//! Deserialize Java NoiseRouterBlobifier binary format → AstNode.
//!
//! Protocol: 计划.txt §4
//!   Header:  magic(u32=0x4D433257 LE)  version(u16=1)  reserved([u8;4])
//!   Tree  :  [tag:u8][payload]  (recursive)
//!   Tail  :  SHAPE_CONFIG  BIOME_SOURCE
//!
//! Byte order: Little Endian.
//!
//! Every tag MUST match NoiseRouterBlobifier.java constants exactly.

use anyhow::anyhow;

use crate::ast::AstNode;
use crate::Result;

/// Parse a full blob including header. Returns the Root node plus shape info.
pub fn deserialize_full(_blob: &[u8]) -> Result<(AstNode, ShapeConfig)> {
    // TODO skeleton:
    //   1. check blob.len() >= 10, else Deserialize("truncated header")
    //   2. read u32 magic at offset 0, assert == 0x4D433257
    //   3. read u16 version at offset 4, assert == 1
    //   4. parse tree from offset 10 via parse_node
    //   5. parse SHAPE_CONFIG = [min_y:i32, height:i32, cell_width:i32, cell_height:i32]
    //   6. parse BIOME_SOURCE remainder
    //   7. return (root, shape_cfg)
    Ok((
        AstNode::Root(vec![]),
        ShapeConfig {
            min_y: -64,
            height: 384,
            cell_width: 4,
            cell_height: 4,
        },
    ))
}

/// Recursively parse one `[tag:u8][payload]` node starting at bytes[offset..].
/// Returns (node, bytes_consumed).
pub fn parse_node(bytes: &[u8], offset: usize) -> Result<(AstNode, usize)> {
    if offset >= bytes.len() {
        return Err(anyhow!("binary blob deserialize error: unexpected EOF"));
    }
    let tag = bytes[offset];
    let mut cursor = offset + 1;
    use AstNode::*;
    let node = match tag {
        0x01 => {
            // Constant(f32) — 4 bytes IEEE 754 LE
            ensure(bytes.len() >= cursor + 4, "Constant")?;
            let bits = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;
            Constant(f32::from_bits(bits))
        }
        0x02 => InputX,
        0x03 => InputY,
        0x04 => InputZ,
        // TODO: implement all remaining tags (0x10..0xFE) recursively
        other => return Err(anyhow!("unsupported AST node tag: 0x{other:02X}")),
    };
    Ok((node, cursor - offset))
}

/// Minimum remaining-bytes check helper.
fn ensure(have: bool, ctx: &str) -> Result<()> {
    if have {
        Ok(())
    } else {
        Err(anyhow!(
            "binary blob deserialize error: truncated payload while parsing {ctx}"
        ))
    }
}

/// MC shape config: min Y, total world height, horizontal & vertical cell size.
#[derive(Debug, Clone, Copy)]
pub struct ShapeConfig {
    pub min_y: i32,
    pub height: i32,
    pub cell_width: i32,
    pub cell_height: i32,
}
