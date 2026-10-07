# Copyright 2026 European Molecular Biology Laboratory
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Tests the confidence dataclasses built from an inference result.

The chain-level metrics are computed from small arrays with the same functions
the model uses, so the tests run on CPU without model parameters.
"""

from collections.abc import Mapping, Sequence
import json

from absl.testing import absltest
from absl.testing import parameterized
from alphafold3.model import confidence_types
from alphafold3.model import confidences
from alphafold3.model import features
from alphafold3.model import model
from alphafold3.model.atom_layout import atom_layout
import numpy as np


def _asym_ids(token_chain_ids: Sequence[str]) -> np.ndarray:
  """Returns per-token asym IDs, assigned as during featurisation."""
  num_tokens = len(token_chain_ids)
  all_tokens = atom_layout.AtomLayout(
      atom_name=np.full(num_tokens, 'CA', dtype=object),
      res_id=np.arange(1, num_tokens + 1),
      chain_id=np.array(token_chain_ids, dtype=object),
      res_name=np.full(num_tokens, 'GLY', dtype=object),
  )
  chains = features._compute_asym_entity_and_sym_id(all_tokens)
  chain_id_to_asym_id = dict(zip(chains.chain_id, chains.asym_id))
  return np.array([chain_id_to_asym_id[c] for c in token_chain_ids])


def _inference_result(
    token_chain_ids: Sequence[str], chain_scores: Mapping[str, float]
) -> model.InferenceResult:
  """Returns an inference result with chain-level metrics as the model has.

  Args:
    token_chain_ids: The chain ID of every token.
    chain_scores: The TM-adjusted PAE of every row of a chain's tokens, which
      makes it that chain's pTM.
  """
  num_tokens = len(token_chain_ids)
  asym_ids = _asym_ids(token_chain_ids)
  mask = np.ones((num_tokens, num_tokens), dtype=bool)
  row_scores = np.array([chain_scores[c] for c in token_chain_ids])
  # [num_samples=1, num_tokens, num_tokens].
  tm_adjusted_pae = np.repeat(row_scores[None, :, None], num_tokens, axis=2)
  chain_pair_iptm = model._compute_chain_pair_iptm(
      num_tokens=num_tokens,
      asym_ids=asym_ids,
      mask=mask,
      tm_adjusted_pae=tm_adjusted_pae,
  )
  _, chain_pair_pae_min, _ = confidences.chain_pair_pae(
      num_tokens=num_tokens,
      asym_ids=asym_ids,
      full_pae=10.0 * (1.0 - tm_adjusted_pae),
      mask=mask,
  )
  return model.InferenceResult(
      predicted_structure=None,  # pyrefly: ignore[bad-argument-type]
      metadata={
          'ptm': 0.5,
          'iptm': 0.5,
          'ranking_score': 0.5,
          'fraction_disordered': 0.0,
          'has_clash': 0.0,
          'chain_pair_pae_min': chain_pair_pae_min[0],
          'chain_pair_iptm': chain_pair_iptm[0],
          'iptm_ichain': chain_pair_iptm[0].diagonal(),
          'iptm_xchain': confidences.get_iptm_xchain(chain_pair_iptm)[0],
          'token_chain_ids': list(token_chain_ids),
      },
  )


class StructureConfidenceSummaryTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='two_chains',
          token_chain_ids='AAABB',
          expected_chain_ids=['A', 'B'],
      ),
      dict(
          testcase_name='first_chain_not_first_alphabetically',
          token_chain_ids='BBBAA',
          expected_chain_ids=['B', 'A'],
      ),
      dict(
          testcase_name='three_chains_with_single_token_chain',
          token_chain_ids='CCABBBB',
          expected_chain_ids=['C', 'A', 'B'],
      ),
  )
  def test_chain_ids_follow_chain_level_arrays(
      self, token_chain_ids, expected_chain_ids
  ):
    chain_scores = {'A': 0.3, 'B': 0.6, 'C': 0.9}
    result = _inference_result(list(token_chain_ids), chain_scores)

    summary = confidence_types.StructureConfidenceSummary.from_inference_result(
        result
    )

    self.assertEqual(summary.chain_ids, expected_chain_ids)
    num_chains = len(expected_chain_ids)
    self.assertLen(summary.chain_ptm, num_chains)
    self.assertLen(summary.chain_iptm, num_chains)
    self.assertEqual(summary.chain_pair_iptm.shape, (num_chains, num_chains))
    self.assertEqual(
        summary.chain_pair_pae_min.shape, (num_chains, num_chains)
    )
    # Each chain's pTM is its own score, so chain_ids[i] labels chain_ptm[i].
    np.testing.assert_allclose(
        summary.chain_ptm,
        [chain_scores[c] for c in summary.chain_ids],
        rtol=1e-6,
    )
    summary_json = json.loads(summary.to_json())
    self.assertEqual(summary_json['chain_ids'], expected_chain_ids)
    self.assertLen(summary_json['chain_ptm'], num_chains)


if __name__ == '__main__':
  absltest.main()
