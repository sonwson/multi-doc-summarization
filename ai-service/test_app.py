import importlib.util
import unittest
from pathlib import Path


APP_PATH = Path(__file__).resolve().with_name("app.py")
SPEC = importlib.util.spec_from_file_location("ai_app_under_test", APP_PATH)
AI_APP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AI_APP)


class ClusterRankMmrModelSmokeTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.model, cls.tokenizer, cls.cfg, cls.checkpoint, cls.checkpoint_path, cls.tokenizer_dir = AI_APP.load_artifacts()

  def test_resolves_latest_results_checkpoint(self):
    expected = Path("phobert_cluster_rank_mmr_outputs/best_model.pt")
    self.assertEqual(self.checkpoint_path.relative_to(AI_APP.ROOT_DIR), expected)
    self.assertEqual(self.checkpoint.get("best_decoder_params"), {
      "MIN_SENT_SCORE": 0.25,
      "SUMMARY_MAX_SENTENCES": 7,
      "MIN_REQUIRED_SENTENCES": 1,
      "SUMMARY_MAX_WORDS": 240,
      "MMR_ALPHA": 0.55,
      "REDUNDANCY_WEIGHT": 0.25,
      "CENTRALITY_WEIGHT": 0.06,
    })
    self.assertEqual(self.cfg.MIN_SENT_SCORE, 0.15)

  def test_can_summarize_with_latest_results_model(self):
    documents = [
      "Tri tue nhan tao dang duoc ung dung rong rai trong giao duc va y te. Nhieu truong hoc su dung cong cu tu dong hoa de ho tro hoc tap. Benh vien cung ung dung he thong phan tich du lieu de ho tro chan doan.",
      "Doanh nghiep tiep tuc dau tu ha tang du lieu va mo hinh ngon ngu de nang cao hieu suat van hanh. Viec quan tri du lieu va danh gia rui ro tro thanh yeu cau quan trong.",
    ]
    result = AI_APP.summarize_documents(documents, self.model, self.tokenizer, self.cfg)

    self.assertTrue(result["summary"].strip())
    self.assertGreaterEqual(len(result["selected_indices"]), 1)
    self.assertEqual(result["decoding_mode"], "cluster_mmr")


if __name__ == "__main__":
  unittest.main()
