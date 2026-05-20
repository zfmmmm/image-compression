from pathlib import Path

from PIL import Image

from compress import parse_bool as parse_compress_bool
from codecs.base import CodecResult
from eval import _compress_one_image
from strategy.adaptive_selector import AdaptiveSelector
from utils.output_artifacts import copy_best_bitstream
from utils.output_options import resolve_output_options


class FakeCodec:
    name = "fake"
    bitstream_extension = ".fake"

    def is_available(self) -> bool:
        return True

    def candidate_params(self) -> list[dict]:
        return [{"quality": 1}]

    def compress_and_eval(self, input_path: Path, work_dir: Path, param: dict) -> CodecResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        bitstream_path = work_dir / "compressed.fake"
        recon_path = work_dir / "recon.png"
        bitstream_path.write_bytes(b"fake-bitstream")
        Image.open(input_path).save(recon_path)
        return CodecResult(
            codec_name=self.name,
            input_path=input_path,
            bitstream_path=bitstream_path,
            recon_path=recon_path,
            param=dict(param),
            original_theoretical_size=768,
            compressed_size=48,
            compression_ratio=16.0,
            bpp=1.5,
            psnr=40.0,
            ssim=0.95,
            ms_ssim=None,
            edge_psnr=39.0,
            encode_time_ms=1.0,
            decode_time_ms=1.0,
            success=True,
            error_message=None,
        )


def _write_test_image(path: Path) -> None:
    image = Image.new("RGB", (16, 16), (64, 128, 192))
    image.save(path)


def test_selector_can_skip_candidate_artifacts_and_all_results_csv(tmp_path):
    input_path = tmp_path / "input.png"
    _write_test_image(input_path)

    selector = AdaptiveSelector(
        [FakeCodec()],
        save_candidate_artifacts=False,
        save_all_results_csv=False,
    )

    best, results, _features = selector.compress(input_path, tmp_path / "run")

    assert len(results) == 1
    assert best.bitstream_path == tmp_path / "run" / "best" / "compressed.fake"
    assert best.bitstream_path.read_bytes() == b"fake-bitstream"
    assert not (tmp_path / "run" / "all_results.csv").exists()
    assert not (tmp_path / "run" / "candidates").exists()


def test_copy_best_bitstream_writes_flat_named_file(tmp_path):
    source = tmp_path / "work" / "best" / "compressed.avif"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"avif-data")
    result = CodecResult(
        codec_name="avif",
        input_path=Path("kodim01.png"),
        bitstream_path=source,
        recon_path=tmp_path / "work" / "best" / "recon.png",
        param={},
        original_theoretical_size=768,
        compressed_size=48,
        compression_ratio=16.0,
        bpp=1.5,
        psnr=40.0,
        ssim=0.95,
        ms_ssim=None,
        edge_psnr=39.0,
        encode_time_ms=1.0,
        decode_time_ms=1.0,
        success=True,
        error_message=None,
    )

    copied_path = copy_best_bitstream(result, tmp_path / "flat", "kodim01")

    assert copied_path == tmp_path / "flat" / "kodim01.avif"
    assert copied_path.read_bytes() == b"avif-data"


def test_eval_worker_can_discard_candidate_results_when_all_results_csv_is_disabled(tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    input_path = input_dir / "input.png"
    _write_test_image(input_path)

    best, candidates = _compress_one_image(
        image_path=input_path,
        input_dir=input_dir,
        output_dir=tmp_path / "report",
        codecs=[FakeCodec()],
        target_ratio=16.0,
        min_psnr=35.0,
        mode="ratio_first",
        max_trials_per_codec=None,
        search_mode="exhaustive",
        save_candidate_artifacts=False,
        save_all_results_csv=False,
        keep_candidate_results=False,
    )

    assert best.codec_name == "fake"
    assert candidates == []
    assert not (tmp_path / "report" / "images" / "input" / "candidates").exists()
    assert not (tmp_path / "report" / "images" / "input" / "all_results.csv").exists()


def test_compress_cli_parse_bool_accepts_readme_values():
    assert parse_compress_bool("true") is True
    assert parse_compress_bool("1") is True
    assert parse_compress_bool("false") is False
    assert parse_compress_bool("0") is False


def test_best_only_output_mode_disables_intermediate_artifacts_and_uses_output_dir():
    options = resolve_output_options(
        output_dir=Path("compressed"),
        best_bitstreams_dir=None,
        best_only=True,
        save_candidates=True,
        save_all_results_csv=True,
    )

    assert options.save_candidate_artifacts is False
    assert options.save_all_results_csv is False
    assert options.best_bitstreams_dir == Path("compressed")
