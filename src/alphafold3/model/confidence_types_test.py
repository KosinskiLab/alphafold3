# Copyright 2024 DeepMind Technologies Limited
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

from collections.abc import Mapping, Sequence
import json
import re

from absl.testing import absltest
from absl.testing import parameterized
from alphafold3 import structure
from alphafold3.model import confidence_types
from alphafold3.model import confidences
from alphafold3.model import features
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model import model
import numpy as np

ConfidenceCategory = confidence_types.ConfidenceCategory


def _get_test_structure() -> structure.Structure:
  return structure.from_atom_arrays(
      res_name=np.array(
          ['A'] * 3 + ['R'] * 2 + ['N'] * 4 + ['R'] * 2 + ['N'] * 2 + ['D'] * 3,
          dtype=object,
      ),
      res_id=np.array(
          [1, 1, 1, 2, 2, 3, 3, 3, 3, 1, 1, 2, 2, 3, 3, 3], dtype=np.int32
      ),
      chain_id=np.array(9 * ['A'] + 7 * ['B'], dtype=object),
      chain_type=np.array(16 * ['polypeptide(L)'], dtype=object),
      atom_b_factor=np.array(
          [70, 80, 90, 10, 30, 50, 50, 50, 50] + 7 * [95], dtype=np.float32
      ),
  )


class AtomConfidenceTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.inference_result = model.InferenceResult(
        predicted_structure=_get_test_structure(),
    )

  def test_instantiation_ok(self):
    confidence_types.AtomConfidence(
        chain_id=['A', 'A'],
        atom_number=[0, 1],
        confidence=[80.0, 60.0],
        confidence_category=[ConfidenceCategory.MEDIUM, ConfidenceCategory.LOW],
    )

  def test_instantiation_error(self):
    with self.assertRaises(ValueError):
      confidence_types.AtomConfidence(
          chain_id=['A', 'A'],
          atom_number=[0, 1, 2],
          confidence=[80.0, 60.0],
          confidence_category=[ConfidenceCategory.MEDIUM],
      )

  def test_from_inference_result_ok(self):
    confidence_1d = confidence_types.AtomConfidence.from_inference_result(
        self.inference_result
    )

    with self.subTest('confidence_type'):
      self.assertIsInstance(confidence_1d.confidence[0], float)
      self.assertIsInstance(confidence_1d.confidence[1], float)
      self.assertIsInstance(confidence_1d.confidence[2], float)

    with self.subTest('confidence_value'):
      self.assertAlmostEqual(confidence_1d.confidence[0], 70.0)
      self.assertAlmostEqual(confidence_1d.confidence[3], 10.0)
      self.assertAlmostEqual(confidence_1d.confidence[6], 50.0)

    with self.subTest('confidence_category'):
      self.assertEqual(
          confidence_1d.confidence_category[0],
          ConfidenceCategory.MEDIUM,
      )
      self.assertEqual(
          confidence_1d.confidence_category[3],
          ConfidenceCategory.DISORDERED,
      )
      self.assertEqual(
          confidence_1d.confidence_category[6],
          ConfidenceCategory.LOW,
      )
      self.assertEqual(
          confidence_1d.confidence_category[10],
          ConfidenceCategory.HIGH,
      )

  def test_serialization_ok(self):
    confidence_1d = confidence_types.AtomConfidence.from_inference_result(
        self.inference_result
    )
    as_json = confidence_1d.to_json()
    from_json = confidence_types.AtomConfidence.from_json(as_json)
    self.assertIsInstance(as_json, str)
    self.assertEqual(from_json, confidence_1d)


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

  def test_from_inference_result_ok(self):
    metadata = {
        'ptm': 0.613,
        'iptm': 0.432,
        'ranking_score': 0.789,
        'fraction_disordered': 0.123,
        'has_clash': True,
        'chain_pair_pae_min': np.array([[0.0, 0.2], [0.1, 0.0]]),
        'chain_pair_iptm': np.array([[0.6, 0.3], [0.2, 0.5]]),
        'iptm_ichain': np.array([0.9, 0.8]),
        'iptm_xchain': np.array([0.6, 0.5]),
        'token_chain_ids': ['A', 'B'],
    }
    inference_result = model.InferenceResult(
        metadata=metadata,  # pyrefly: ignore[bad-argument-type]
        predicted_structure=_get_test_structure(),
    )
    structure_confidence = (
        confidence_types.StructureConfidenceSummary.from_inference_result(
            inference_result
        )
    )
    np.testing.assert_array_equal(structure_confidence.ptm, 0.613)
    np.testing.assert_array_equal(structure_confidence.iptm, 0.432)
    np.testing.assert_array_equal(structure_confidence.ranking_score, 0.789)
    np.testing.assert_array_equal(
        structure_confidence.fraction_disordered, 0.123
    )
    np.testing.assert_array_equal(structure_confidence.has_clash, 1.0)
    np.testing.assert_array_equal(
        structure_confidence.chain_pair_pae_min,
        [[0.0, 0.2], [0.1, 0.0]],
    )
    np.testing.assert_array_equal(
        structure_confidence.chain_pair_iptm,
        [[0.6, 0.3], [0.2, 0.5]],
    )
    np.testing.assert_array_equal(structure_confidence.chain_ptm, [0.9, 0.8])
    np.testing.assert_array_equal(structure_confidence.chain_iptm, [0.6, 0.5])
    self.assertEqual(structure_confidence.chain_ids, ['A', 'B'])

  @parameterized.named_parameters(
      ('float32', np.float32), ('float64', np.float64)
  )
  def test_serialization_format(self, dtype):
    structure_confidence = confidence_types.StructureConfidenceSummary(
        ptm=0.612,
        iptm=0.432,
        ranking_score=0.789,
        fraction_disordered=0.0,
        has_clash=0.0,
        chain_pair_pae_min=np.array([[0.0, 0.2], [0.1, 0.0]], dtype=dtype),
        chain_pair_iptm=np.array([[0.6, 0.3], [0.2, 0.5]], dtype=dtype),
        chain_ptm=np.array([0.8, 0.9], dtype=dtype),
        chain_iptm=np.array([0.6, 0.5], dtype=dtype),
    )
    as_json = structure_confidence.to_json()
    # Check that all floats in json string have at most 2 decimal digit.
    regex_all_floats = r'\d+\.?\d*'
    regex_floats_with_3_decimals_or_more = r'\d+\.?\d\d\d+'
    self.assertLen(re.findall(regex_all_floats, as_json), 17)
    self.assertEmpty(re.findall(regex_floats_with_3_decimals_or_more, as_json))

  def test_serialization_ok(self):
    structure_confidence = confidence_types.StructureConfidenceSummary(
        ptm=0.612,
        iptm=0.432,
        ranking_score=0.789,
        fraction_disordered=0.0,
        has_clash=0.0,
        chain_pair_pae_min=np.array([[0.0, 0.2], [0.1, 0.0]]),
        chain_pair_iptm=np.array([[0.6, 0.3], [0.2, 0.5]]),
        chain_ptm=np.array([0.8, 0.9]),
        chain_iptm=np.array([0.6, 0.5]),
    )
    as_json = structure_confidence.to_json()
    from_json = confidence_types.StructureConfidenceSummary.from_json(as_json)
    self.assertIsInstance(as_json, str)
    # Check equality up to 2 decimal, which is how much we round values when
    # serializing to json.
    np.testing.assert_array_almost_equal(
        from_json.ptm,
        structure_confidence.ptm,
        decimal=2,
    )
    np.testing.assert_array_almost_equal(
        from_json.iptm,
        structure_confidence.iptm,
        decimal=2,
    )
    np.testing.assert_array_almost_equal(
        from_json.ranking_score,
        structure_confidence.ranking_score,
        decimal=2,
    )


class StructureConfidenceFullTest(parameterized.TestCase):

  def test_from_inference_result_ok(self):
    metadata = {
        'ptm': 0.613,
        'iptm': 0.432,
        'ranking_score': 0.789,
        'fraction_disordered': 0.0,
        'has_clash': 0.0,
        'token_chain_ids': ['A', 'A', 'B', 'B', 'C', 'D'],
        'token_res_ids': [1, 2, 1, 2, 1, 1],
    }
    numerical_data = {
        'full_pae': np.zeros((6, 6)),
        'contact_probs': np.zeros((6, 6)),
    }
    inference_result = model.InferenceResult(
        metadata=metadata,  # pyrefly: ignore[bad-argument-type]
        numerical_data=numerical_data,
        predicted_structure=_get_test_structure(),
    )
    structure_confidence = (
        confidence_types.StructureConfidenceFull.from_inference_result(
            inference_result
        )
    )
    self.assertLen(structure_confidence.pae, 6)
    self.assertLen(structure_confidence.pae[0], 6)
    self.assertLen(structure_confidence.contact_probs, 6)
    self.assertLen(structure_confidence.contact_probs[0], 6)
    np.testing.assert_array_equal(
        structure_confidence.atom_plddts,
        [70, 80, 90, 10, 30, 50, 50, 50, 50, 95, 95, 95, 95, 95, 95, 95],
    )
    np.testing.assert_array_equal(
        structure_confidence.atom_chain_ids, ['A'] * 9 + ['B'] * 7
    )
    np.testing.assert_array_equal(
        structure_confidence.token_chain_ids, ['A', 'A', 'B', 'B', 'C', 'D']
    )
    np.testing.assert_array_equal(
        structure_confidence.token_res_ids, [1, 2, 1, 2, 1, 1]
    )

  @parameterized.named_parameters(
      ('float32', np.float32), ('float64', np.float64)
  )
  def test_serialization_format(self, dtype):
    structure_confidence = confidence_types.StructureConfidenceFull(
        pae=np.array([[0.123, 0.456], [0.78901, 0.0]], dtype),
        token_chain_ids=['A', 'B'],
        token_res_ids=[1, 1],
        atom_plddts=[0.432, 0.4321],
        atom_chain_ids=['A', 'B'],
        contact_probs=np.array([[0.01111, 0.2222], [0.00333, 0.4]], dtype),
    )
    as_json = structure_confidence.to_json()
    # Check that all floats in json string have at most 2 decimal digits.
    regex_all_floats = r'\d+\.?\d*'
    regex_floats_with_3_decimals_or_more = r'\d+\.?\d\d\d+'
    self.assertLen(re.findall(regex_all_floats, as_json), 12)
    self.assertEmpty(re.findall(regex_floats_with_3_decimals_or_more, as_json))

  def test_serialization_ok(self):
    structure_confidence = confidence_types.StructureConfidenceFull(
        atom_chain_ids=['A', 'B'],
        atom_plddts=[0.432, 0.4321],
        contact_probs=np.array([[0.01111, 0.2222], [0.00333, 0.4]]),
        pae=np.array([[0.123, 0.456], [0.78901, 0.0]]),
        token_chain_ids=['A', 'B'],
        token_res_ids=[1, 1],
    )
    as_json = structure_confidence.to_json()
    from_json = confidence_types.StructureConfidenceFull.from_json(as_json)
    self.assertIsInstance(as_json, str)
    np.testing.assert_array_almost_equal(
        from_json.pae,
        structure_confidence.pae,
        decimal=1,
    )
    np.testing.assert_array_equal(
        from_json.token_chain_ids,
        structure_confidence.token_chain_ids,
    )
    np.testing.assert_array_equal(
        from_json.token_res_ids,
        structure_confidence.token_res_ids,
    )
    np.testing.assert_array_almost_equal(
        from_json.contact_probs,
        structure_confidence.contact_probs,
        decimal=2,
    )
    np.testing.assert_array_almost_equal(
        from_json.atom_plddts,
        structure_confidence.atom_plddts,
        decimal=2,
    )
    np.testing.assert_array_equal(
        from_json.atom_chain_ids,
        structure_confidence.atom_chain_ids,
    )

  def test_serialization_nan(self):
    structure_confidence = confidence_types.StructureConfidenceFull(
        pae=np.array([[0.123, 0.456], [0.78901, float('NaN')]]),
        token_chain_ids=['A', 'B'],
        token_res_ids=[1, 1],
        atom_plddts=[0.432, float('nan')],
        atom_chain_ids=['A', 'B'],
        contact_probs=np.array([[0.01111, 0.2222], [0.00333, 0.4]]),
    )
    json = structure_confidence.to_json()
    self.assertNotIn('nan', json)
    self.assertNotIn('NaN', json)
    self.assertIn('null', json)


class ConfidenceCategoryTest(parameterized.TestCase):

  @parameterized.parameters([
      ('H', ConfidenceCategory.HIGH),
      ('M', ConfidenceCategory.MEDIUM),
      ('L', ConfidenceCategory.LOW),
      ('D', ConfidenceCategory.DISORDERED),
  ])
  def test_from_char_ok(self, char: str, expected: ConfidenceCategory):
    self.assertEqual(ConfidenceCategory.from_char(char), expected)

  @parameterized.parameters(['h', 'm', 'l', 'd', '', 'U', 'A'])
  def test_from_char_error(self, char: str):
    with self.assertRaises(ValueError):
      ConfidenceCategory.from_char(char)

  @parameterized.parameters([
      (ConfidenceCategory.HIGH, 'H'),
      (ConfidenceCategory.MEDIUM, 'M'),
      (ConfidenceCategory.LOW, 'L'),
      (ConfidenceCategory.DISORDERED, 'D'),
  ])
  def test_to_char_ok(self, category: ConfidenceCategory, char: str):
    self.assertEqual(category.to_char(), char)

  @parameterized.parameters([
      (95, ConfidenceCategory.HIGH),
      (85, ConfidenceCategory.MEDIUM),
      (60, ConfidenceCategory.LOW),
      (30, ConfidenceCategory.DISORDERED),
  ])
  def test_from_confidence_score_ok(
      self, score: float, expected: ConfidenceCategory
  ):
    self.assertEqual(ConfidenceCategory.from_confidence_score(score), expected)

  def test_from_confidence_score_error(self):
    with self.assertRaises(ValueError):
      ConfidenceCategory.from_confidence_score(-1)
    with self.assertRaises(ValueError):
      ConfidenceCategory.from_confidence_score(101)


if __name__ == '__main__':
  absltest.main()
