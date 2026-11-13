"""Tests for zip traversal and page-image extraction.

Uses one real source PDF as a fixture (AC1 part 3, the smallest in the corpus) so the
geometry assertions are checked against genuine publisher output rather than a mock.
"""

import shutil
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from assam_rolls import render

FIXTURE_PDF = (
    Path(__file__).parent
    / "fixtures"
    / ("2026-EROLLGEN-S03-1-FinalRoll-Revision1-ASM-3-WI_INFO.pdf")
)

needs_poppler = pytest.mark.skipif(
    shutil.which("pdfimages") is None, reason="poppler not installed"
)


@pytest.fixture
def zip_with_fixture(tmp_path: Path) -> Path:
    """A zip shaped like a real AC bundle, plus one file the parser must ignore."""
    zip_path = tmp_path / "AC1_ASM Roll Info Pages.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(FIXTURE_PDF, FIXTURE_PDF.name)
        archive.writestr("readme.txt", "not a pdf")
        archive.writestr("unexpected-name.pdf", "%PDF-1.7 stub")
    return zip_path


class TestIterZipParts:
    def test_finds_the_roll_pdf_and_skips_the_rest(self, zip_with_fixture):
        refs = list(render.iter_zip_parts(zip_with_fixture))
        assert len(refs) == 1
        assert refs[0].ac_no == 1
        assert refs[0].part_no == 3
        assert refs[0].zip_name == zip_with_fixture.name

    def test_reports_unrecognized_pdfs(self, zip_with_fixture):
        """Silently skipping an unexpected PDF would hide missing data."""
        assert render.unrecognized_pdfs(zip_with_fixture) == ["unexpected-name.pdf"]

    def test_sorts_numerically_not_lexicographically(self, tmp_path):
        zip_path = tmp_path / "AC1_ASM Roll Info Pages.zip"
        template = "2026-EROLLGEN-S03-1-FinalRoll-Revision1-ASM-{}-WI_INFO.pdf"
        with zipfile.ZipFile(zip_path, "w") as archive:
            for part in (154, 9, 99, 1):
                archive.writestr(template.format(part), "%PDF-1.7 stub")
        assert [r.part_no for r in render.iter_zip_parts(zip_path)] == [1, 9, 99, 154]


class TestPartRef:
    def test_key_is_zero_padded_and_sortable(self):
        ref = render.PartRef("z.zip", "a.pdf", ac_no=1, part_no=3)
        assert ref.key == "001-0003"

    def test_keys_sort_in_numeric_order(self):
        keys = [render.PartRef("z", "a", ac_no=1, part_no=part).key for part in (154, 9, 99, 1)]
        assert sorted(keys) == ["001-0001", "001-0009", "001-0099", "001-0154"]


class TestSha256:
    def test_is_stable_and_content_dependent(self):
        assert render.sha256_bytes(b"abc") == render.sha256_bytes(b"abc")
        assert render.sha256_bytes(b"abc") != render.sha256_bytes(b"abd")


@needs_poppler
class TestExtractPageImage:
    def test_page_one_matches_native_geometry(self):
        image = render.extract_page_image(FIXTURE_PDF.read_bytes(), page=render.PAGE_FORM)
        assert image.size == render.EXPECTED_PAGE_SIZE

    def test_page_two_is_also_extractable(self):
        image = render.extract_page_image(FIXTURE_PDF.read_bytes(), page=render.PAGE_IMAGERY)
        assert image.size == render.EXPECTED_PAGE_SIZE

    def test_extraction_does_not_resample(self):
        """Byte-exact extraction is the whole point; upsampling would inflate token cost."""
        image = render.extract_page_image(FIXTURE_PDF.read_bytes())
        assert image.size == render.EXPECTED_PAGE_SIZE
        assert image.mode == "RGB"

    def test_raises_on_a_non_pdf(self):
        with pytest.raises(render.RenderError):
            render.extract_page_image(b"not a pdf at all")


class TestFormCrop:
    def test_keeps_full_width_and_trims_height(self):
        image = Image.new("RGB", render.EXPECTED_PAGE_SIZE)
        cropped = render.form_crop(image)
        assert cropped.size == (1187, int(round(1679 * render.FORM_CROP_FRACTION)))

    def test_fraction_is_configurable(self):
        image = Image.new("RGB", (100, 200))
        assert render.form_crop(image, fraction=0.5).size == (100, 100)


class TestEncodePng:
    def test_round_trips(self):
        image = Image.new("RGB", (10, 20), color=(1, 2, 3))
        with Image.open(__import__("io").BytesIO(render.encode_png(image))) as decoded:
            assert decoded.size == (10, 20)


@needs_poppler
class TestRenderZip:
    def test_writes_one_png_per_part(self, zip_with_fixture, tmp_path):
        out = tmp_path / "pages"
        refs = render.render_zip(zip_with_fixture, out)
        assert len(refs) == 1
        assert (out / "001-0003.png").exists()

    def test_is_resumable(self, zip_with_fixture, tmp_path):
        """A second run must not redo work -- statewide this is 28k pages."""
        out = tmp_path / "pages"
        render.render_zip(zip_with_fixture, out)
        target = out / "001-0003.png"
        target.write_bytes(b"sentinel")
        render.render_zip(zip_with_fixture, out)
        assert target.read_bytes() == b"sentinel"

    def test_overwrite_forces_rerender(self, zip_with_fixture, tmp_path):
        out = tmp_path / "pages"
        render.render_zip(zip_with_fixture, out)
        target = out / "001-0003.png"
        target.write_bytes(b"sentinel")
        render.render_zip(zip_with_fixture, out, overwrite=True)
        assert target.read_bytes() != b"sentinel"

    def test_limit_caps_work(self, zip_with_fixture, tmp_path):
        assert render.render_zip(zip_with_fixture, tmp_path / "pages", limit=0) == []


class TestNormalizePage:
    """ACs 64-73 are issued at 949x1343 rather than 1187x1679 -- a uniform 0.80x.

    Every pixel constant downstream assumes the canonical size, so an un-normalised page
    fails grid detection outright: 1,554 rows were flagged before the block finished.
    """

    def test_canonical_pages_are_returned_untouched(self):
        """The common path must not resample: most pages are already the right size."""
        page = Image.new("RGB", render.CANONICAL_PAGE_SIZE, "white")
        assert render.normalize_page(page) is page

    def test_a_smaller_page_is_scaled_up(self):
        page = Image.new("RGB", (949, 1343), "white")
        assert render.normalize_page(page).size == render.CANONICAL_PAGE_SIZE

    def test_a_larger_page_is_scaled_down(self):
        page = Image.new("RGB", (2374, 3358), "white")
        assert render.normalize_page(page).size == render.CANONICAL_PAGE_SIZE

    def test_the_observed_variant_is_a_uniform_scale(self):
        """Both axes scale by the same factor, so the form is not distorted."""
        wide, high = 949 / render.CANONICAL_PAGE_SIZE[0], 1343 / render.CANONICAL_PAGE_SIZE[1]
        assert abs(wide - high) < 0.001, "a non-uniform scale would need more than a resize"


class TestResamplingFilter:
    """The filter is load-bearing, not cosmetic.

    Measured over 30 affected pages, LANCZOS read start_serial correctly on 0 of 30 and
    NEAREST on 30 of 30: its ringing thickens the thin "1" until Tesseract calls it "4".
    The elector sum does not involve the start serial, so the error passed every check --
    LANCZOS balanced 30/30 while being wrong on all 30.
    """

    def test_uses_a_non_interpolating_filter(self):
        """Interpolation invents pixels, and invented pixels look like ink."""
        assert render.RESAMPLING is Image.Resampling.NEAREST

    def test_a_thin_stroke_survives_upscaling(self):
        """A one-pixel vertical stroke must not gain neighbors it never had."""
        small = Image.new("L", (20, 20), 255)
        for y in range(4, 16):
            small.putpixel((10, y), 0)
        big = small.resize((40, 40), render.RESAMPLING)
        # every pixel is either the original ink or the original paper, nothing between
        assert set(big.tobytes()) <= {0, 255}
