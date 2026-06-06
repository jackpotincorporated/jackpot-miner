//! Keyed-BLAKE3 Merkle tree + multi-leaf proof — port of the reference
//! `pearl-blake3` (ISC), built on the stock `blake3::hazmat` API so the tree's
//! node hashes (and root) are bit-identical to BLAKE3's own tree. Used to build
//! the `PlainProof` Merkle authentication paths for share submission.

use blake3::hazmat::{merge_subtrees_non_root, merge_subtrees_root, HasherExt, Mode};
use blake3::{Hasher, CHUNK_LEN, OUT_LEN};
use std::collections::BTreeSet;

pub type Digest = [u8; OUT_LEN]; // 32

fn chunk_cv(key: &[u8; 32], data: &[u8], idx: u64) -> Digest {
    Hasher::new_keyed(key)
        .set_input_offset(idx * CHUNK_LEN as u64)
        .update(data)
        .finalize_non_root()
}

/// Round `raw_len` up to a multiple of the BLAKE3 chunk (1024).
pub fn padded_chunk_len(raw_len: usize) -> usize {
    raw_len.div_ceil(CHUNK_LEN) * CHUNK_LEN
}

/// Merge precomputed chunk chaining values (`finalize_non_root` CVs, in order)
/// into the keyed-BLAKE3 root — the same pairwise reduction as [`MerkleTree::new`],
/// but for CVs computed elsewhere (e.g. on the GPU). Requires `cvs.len() >= 2`.
pub fn root_from_cvs(mut layer: Vec<Digest>, key: &[u8; 32]) -> Digest {
    let mode = Mode::KeyedHash(key);
    assert!(layer.len() >= 2, "root_from_cvs needs >= 2 chunks");
    while layer.len() > 2 {
        layer = layer
            .chunks(2)
            .map(|p| if p.len() == 2 { merge_subtrees_non_root(&p[0], &p[1], mode) } else { p[0] })
            .collect();
    }
    *merge_subtrees_root(&layer[0], &layer[1], mode).as_bytes()
}

pub fn pad_to_chunk_boundary(data: &[u8]) -> Vec<u8> {
    let mut v = data.to_vec();
    v.resize(padded_chunk_len(v.len()), 0);
    v
}

/// Keyed-BLAKE3 Merkle tree over 1024-byte leaves.
pub struct MerkleTree {
    layers: Vec<Vec<Digest>>,
    data: Vec<u8>,
    key: [u8; 32],
}

impl MerkleTree {
    pub fn new(data: &[u8], key: [u8; 32]) -> Self {
        let mode = Mode::KeyedHash(&key);
        if data.is_empty() {
            return Self { layers: vec![vec![]], data: vec![], key };
        }
        if data.len() <= CHUNK_LEN {
            let root = *blake3::keyed_hash(&key, data).as_bytes();
            return Self { layers: vec![vec![root]], data: data.to_vec(), key };
        }
        let n = data.len().div_ceil(CHUNK_LEN);
        let cvs: Vec<Digest> = (0..n)
            .map(|i| {
                let s = i * CHUNK_LEN;
                let e = (s + CHUNK_LEN).min(data.len());
                chunk_cv(&key, &data[s..e], i as u64)
            })
            .collect();
        let mut layers = vec![cvs];
        while layers.last().unwrap().len() > 2 {
            let prev = layers.last().unwrap();
            let next: Vec<Digest> = prev
                .chunks(2)
                .map(|p| if p.len() == 2 { merge_subtrees_non_root(&p[0], &p[1], mode) } else { p[0] })
                .collect();
            layers.push(next);
        }
        let last = layers.last().unwrap();
        if last.len() == 2 {
            let root = *merge_subtrees_root(&last[0], &last[1], mode).as_bytes();
            layers.push(vec![root]);
        }
        Self { layers, data: data.to_vec(), key }
    }

    pub fn root(&self) -> Digest {
        self.layers.last().map(|l| l[0]).unwrap_or([0u8; OUT_LEN])
    }
    pub fn num_leaves(&self) -> usize {
        self.layers[0].len()
    }

    /// Leaf indices (1024-byte chunks) needed to authenticate the given matrix rows.
    pub fn compute_leaf_indices_from_rows(row_indices: &[usize], shape: (usize, usize)) -> Vec<usize> {
        let cols = shape.1;
        let mut idx = BTreeSet::new();
        for &row in row_indices {
            let first = (row * cols) / CHUNK_LEN;
            let last = ((row + 1) * cols - 1) / CHUNK_LEN;
            for i in first..=last {
                idx.insert(i);
            }
        }
        idx.into_iter().collect()
    }

    pub fn get_multileaf_proof(&self, leaf_indices: &[usize]) -> MerkleProof {
        assert!(!leaf_indices.is_empty());
        let unique: BTreeSet<usize> = leaf_indices.iter().copied().collect();
        let total_leaves = self.num_leaves();
        assert!(*unique.last().unwrap() < total_leaves, "leaf index out of bounds");
        let sorted: Vec<usize> = unique.iter().copied().collect();
        let leaf_data: Vec<[u8; CHUNK_LEN]> = sorted
            .iter()
            .map(|&i| {
                let s = i * CHUNK_LEN;
                let e = (s + CHUNK_LEN).min(self.data.len());
                let mut c = [0u8; CHUNK_LEN];
                c[..e - s].copy_from_slice(&self.data[s..e]);
                c
            })
            .collect();
        let mut siblings = Vec::new();
        let mut cur = unique;
        let mut level_len = total_leaves;
        let mut level = 0;
        while level_len > 1 && !cur.is_empty() {
            let nodes = &self.layers[level];
            for &i in &cur {
                if i % 2 == 1 {
                    if !cur.contains(&(i - 1)) {
                        siblings.push(nodes[i - 1]);
                    }
                } else if !cur.contains(&(i + 1)) && (i + 1) < level_len {
                    siblings.push(nodes[i + 1]);
                }
            }
            cur = cur.iter().map(|&i| i / 2).collect();
            level_len = level_len.div_ceil(2);
            level += 1;
        }
        MerkleProof { leaf_data, leaf_indices: sorted, total_leaves, root: self.root(), siblings }
    }

    #[cfg(test)]
    fn key_used(&self) -> [u8; 32] {
        self.key
    }
}

/// Multi-leaf Merkle proof. `leaf_data` are the raw 1024-byte chunks; `siblings`
/// the authentication path.
#[derive(Clone)]
pub struct MerkleProof {
    pub leaf_data: Vec<[u8; CHUNK_LEN]>,
    pub leaf_indices: Vec<usize>,
    pub total_leaves: usize,
    pub root: Digest,
    pub siblings: Vec<Digest>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_equals_keyed_blake3() {
        let key = [99u8; 32];
        for len in [500usize, 1024, 1500, 4096, 5000] {
            let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            let padded = pad_to_chunk_boundary(&data);
            let tree = MerkleTree::new(&padded, key);
            assert_eq!(tree.root(), *blake3::keyed_hash(&key, &padded).as_bytes(), "len {len}");
            assert_eq!(tree.key_used(), key);
        }
    }

    #[test]
    fn multileaf_proof_collects_siblings() {
        let key = [7u8; 32];
        let data: Vec<u8> = (0..8 * CHUNK_LEN).map(|i| (i % 251) as u8).collect();
        let tree = MerkleTree::new(&data, key);
        let proof = tree.get_multileaf_proof(&[0, 3]);
        assert_eq!(proof.total_leaves, 8);
        assert_eq!(proof.leaf_indices, vec![0, 3]);
        assert!(!proof.siblings.is_empty());
        assert_eq!(proof.root, tree.root());
    }
}
