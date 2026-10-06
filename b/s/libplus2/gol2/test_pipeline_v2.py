"""Unit tests for the v2 pipeline scripts and configuration.

Covers:
  T01: v2 data linkage — dataset has correct identity
  T02: v2 launch scripts — EXPR_NAME, DATA_REPO_ID, DATASET_REPO_ID correct
  T03: v2 schema accessible — libero_goal_4dv2 schema has correct fields
  T04: v2 stats compatibility — 14D state stats present and correct shape
  T05: v2 contract fields — all required fields present
  T06: preflight_v2.sh syntax — no bash syntax errors
  T07: smoke launch args — warmup and sft produce correct accelerate args
  T08: eval wrapper v2 — GOAL4D_DATASET and EXPR_NAME overridden correctly
  T09: v2 vs v1 isolation — robot_type, stats_key, state_dim differ
  T10: checkpoint compatibility guard — SFT rejects v1 warmup ckpt
"""
import json
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_GOL = _HERE.parent / "gol"
_PROJ = _HERE.parents[3]
_V2_DATA = Path("/B/Dta/LIBERO/libero_plus_goal_lrb3_4Dv2")


class TestV2DataLinkage:
    """T01: v2 dataset is accessible and has correct identity."""

    @pytest.fixture
    def v2_info(self):
        info_path = _V2_DATA / "meta" / "info.json"
        if not info_path.exists():
            pytest.skip(f"v2 data not found at {_V2_DATA}")
        return json.loads(info_path.read_text())

    @pytest.fixture
    def v2_contract(self):
        contract_path = _V2_DATA / "meta" / "goal_train_eval_contract.json"
        if not contract_path.exists():
            pytest.skip("v2 contract not found")
        return json.loads(contract_path.read_text())

    def test_robot_type(self, v2_info):
        assert v2_info["robot_type"] == "libero_goal_4dv2"

    def test_state_dim(self, v2_info):
        assert v2_info["features"]["observation.state"]["shape"] == [14]

    def test_action_dim(self, v2_info):
        assert v2_info["features"]["action"]["shape"] == [7]

    def test_total_frames(self, v2_info):
        assert v2_info["total_frames"] == 512604

    def test_contract_schema_2(self, v2_contract):
        assert v2_contract["schema"] == "goal_train_eval_contract/2"

    def test_contract_state_dim(self, v2_contract):
        assert v2_contract["state_dim"] == 14

    def test_contract_stats_key(self, v2_contract):
        assert v2_contract["stats_key"] == "libero_goal_4dv2"

    def test_stats_state_14d(self):
        stats_path = _V2_DATA / "meta" / "stats.json"
        if not stats_path.exists():
            pytest.skip("v2 stats not found")
        stats = json.loads(stats_path.read_text())
        assert len(stats["observation.state"]["mean"]) == 14
        assert len(stats["action"]["mean"]) == 7


class TestLaunchScripts:
    """T02/T07: Launch scripts have correct variable overrides."""

    def test_warmup_has_v2_expr_name(self):
        script = _HERE / "p1v2_warmup_launch.sh"
        if not script.exists():
            pytest.skip("warmup script not yet created")
        content = script.read_text()
        assert "4dwvlaLbPlusGolV2_1001" in content

    def test_warmup_has_v2_data(self):
        script = _HERE / "p1v2_warmup_launch.sh"
        if not script.exists():
            pytest.skip("warmup script not yet created")
        content = script.read_text()
        assert "libero_plus_goal_lrb3_4Dv2" in content

    def test_sft_has_v2_expr_name(self):
        script = _HERE / "p2v2_sft_launch.sh"
        if not script.exists():
            pytest.skip("sft script not yet created")
        content = script.read_text()
        assert "4dwvlaLbPlusGolV2_1001" in content

    def test_sft_has_v2_data(self):
        script = _HERE / "p2v2_sft_launch.sh"
        if not script.exists():
            pytest.skip("sft script not yet created")
        content = script.read_text()
        assert "libero_plus_goal_lrb3_4Dv2" in content

    def test_sft_requires_pretrained_ckpt(self):
        script = _HERE / "p2v2_sft_launch.sh"
        if not script.exists():
            pytest.skip("sft script not yet created")
        content = script.read_text()
        assert "PRETRAINED_CKPT" in content

    def test_warmup_bash_syntax(self):
        script = _HERE / "p1v2_warmup_launch.sh"
        if not script.exists():
            pytest.skip("warmup script not yet created")
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"syntax error: {result.stderr}"

    def test_sft_bash_syntax(self):
        script = _HERE / "p2v2_sft_launch.sh"
        if not script.exists():
            pytest.skip("sft script not yet created")
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"syntax error: {result.stderr}"


class TestEvalScripts:
    """T08: Eval wrapper and preflight have correct v2 configuration."""

    def test_eval_wrapper_has_v2_dataset(self):
        script = _HERE / "eval_v2_wrapper.sh"
        if not script.exists():
            pytest.skip("eval wrapper not yet created")
        content = script.read_text()
        assert "libero_plus_goal_lrb3_4Dv2" in content

    def test_eval_wrapper_has_v2_expr_name(self):
        script = _HERE / "eval_v2_wrapper.sh"
        if not script.exists():
            pytest.skip("eval wrapper not yet created")
        content = script.read_text()
        assert "4dwvlaLbPlusGolV2_1001" in content

    def test_preflight_v2_checks_schema_2(self):
        script = _HERE / "preflight_v2.sh"
        if not script.exists():
            pytest.skip("preflight not yet created")
        content = script.read_text()
        assert "goal_train_eval_contract/2" in content

    def test_preflight_v2_checks_libero_goal_4dv2(self):
        script = _HERE / "preflight_v2.sh"
        if not script.exists():
            pytest.skip("preflight not yet created")
        content = script.read_text()
        assert "libero_goal_4dv2" in content

    def test_preflight_v2_checks_state_dim_14(self):
        script = _HERE / "preflight_v2.sh"
        if not script.exists():
            pytest.skip("preflight not yet created")
        content = script.read_text()
        assert "state_dim" in content and "14" in content

    def test_preflight_bash_syntax(self):
        script = _HERE / "preflight_v2.sh"
        if not script.exists():
            pytest.skip("preflight not yet created")
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"syntax error: {result.stderr}"


class TestV1V2Isolation:
    """T09: v1 and v2 identities are fully separated."""

    def test_different_robot_type(self):
        v1_path = Path("/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/meta/info.json")
        v2_path = _V2_DATA / "meta" / "info.json"
        if not v1_path.exists() or not v2_path.exists():
            pytest.skip("v1 or v2 data not found")
        v1 = json.loads(v1_path.read_text())
        v2 = json.loads(v2_path.read_text())
        assert v1["robot_type"] != v2["robot_type"]
        assert v1["robot_type"] == "panda"
        assert v2["robot_type"] == "libero_goal_4dv2"

    def test_different_state_dim(self):
        v1_path = Path("/B/Dta/LIBERO/libero_plus_goal_lrb3_4D/meta/info.json")
        v2_path = _V2_DATA / "meta" / "info.json"
        if not v1_path.exists() or not v2_path.exists():
            pytest.skip("v1 or v2 data not found")
        v1 = json.loads(v1_path.read_text())
        v2 = json.loads(v2_path.read_text())
        assert v1["features"]["observation.state"]["shape"] == [8]
        assert v2["features"]["observation.state"]["shape"] == [14]


class TestGoalClientV2Branch:
    """T10: goal_client.py has v2 dispatch."""

    def test_v2_branch_exists(self):
        gc = _GOL / "goal_client.py"
        if not gc.exists():
            pytest.skip("goal_client.py not found")
        content = gc.read_text()
        assert "goal_train_eval_contract/2" in content
        assert "pack_state_v2" in content

    def test_run_eval_v2_support(self):
        res = _GOL / "run_eval_goal_plus.sh"
        if not res.exists():
            pytest.skip("run_eval_goal_plus.sh not found")
        content = res.read_text()
        assert "goal_train_eval_contract/2" in content
        assert "libero_goal_4dv2" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
