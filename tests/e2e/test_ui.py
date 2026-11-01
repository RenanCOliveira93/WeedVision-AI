"""E2E com Playwright: sobe a API (uvicorn) e o frontend compilado (vite preview). Webcam = dispositivo FALSO do Chromium (vídeo de teste, não é câmera real).
Executar: pytest tests/e2e -m e2e   (requer `npm run build` em frontend/ e navegador do Playwright instalado)."""
import os, subprocess, sys, time, urllib.request
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "docs/img"
SAMPLE = sorted((ROOT / "sample_data").glob("sudars_test_1*.jpg"))[0]
pytestmark = pytest.mark.e2e


def _wait(url, t=60):
    for _ in range(t * 2):
        try:
            urllib.request.urlopen(url, timeout=2); return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"timeout esperando {url}")


@pytest.fixture(scope="module")
def servers():
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", "8011"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    web = subprocess.Popen(["npx", "vite", "preview", "--host", "127.0.0.1", "--port", "4183", "--strictPort"], cwd=ROOT / "frontend", env={**os.environ, "VITE_API_TARGET": "http://127.0.0.1:8011"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        _wait("http://127.0.0.1:8011/health"); _wait("http://127.0.0.1:4183/")
        yield "http://127.0.0.1:4183"
    finally:
        api.terminate(); web.terminate()


@pytest.fixture(scope="module")
def browser():
    with pw.sync_playwright() as p:
        b = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        yield b
        b.close()


@pytest.fixture
def page(browser, servers):
    ctx = browser.new_context(viewport={"width": 1280, "height": 900}, permissions=["camera"])
    pg = ctx.new_page(); pg.base = servers
    yield pg
    ctx.close()


def test_inferencia_por_upload(page):
    page.goto(page.base)
    assert "não recomenda pulverização" in page.inner_text("header").lower()
    page.set_input_files("input[aria-label='Enviar imagem']", str(SAMPLE))
    page.wait_for_selector("[data-testid=count]", timeout=30000)
    assert int(page.inner_text("[data-testid=count]")) > 0
    assert "NÃO é recomendação de pulverização" in page.inner_text("main")
    page.screenshot(path=str(IMG / "ui_inferencia.png"), full_page=True)


def test_webcam_falsa_infere_quadros(page):
    page.goto(page.base)
    page.click("text=Ligar webcam")
    page.wait_for_selector("[data-testid=count]", timeout=30000)
    assert "inferência" in page.inner_text("main")
    page.click("text=Desligar webcam")


def test_geojson_sem_pose_e_recusado(page):
    page.goto(page.base)
    page.set_input_files("input[aria-label='Enviar imagem']", str(SAMPLE))
    page.wait_for_selector("[data-testid=count]", timeout=30000)
    page.click("text=Exportar GeoJSON")
    page.click("text=Gerar GeoJSON")
    page.wait_for_selector("[role=status]")
    assert "422" in page.inner_text("[role=status]") and "calibração" in page.inner_text("[role=status]")


def test_avaliacao_mostra_dominios_e_classificacao(page):
    page.goto(page.base)
    page.click("role=tab[name='Avaliação e comparação']")
    page.wait_for_selector("table")
    txt = page.inner_text("main")
    assert "outro domínio (zero-shot)" in txt and "mesmo domínio" in txt
    page.click("text=detalhes >> nth=0")
    page.wait_for_selector("svg[aria-label^='Curva']")
    page.screenshot(path=str(IMG / "ui_avaliacao.png"), full_page=True)
    page.click("role=tab[name='Classificação (DeepWeeds)']")
    page.wait_for_selector("text=não há caixas")
    page.screenshot(path=str(IMG / "ui_classificacao.png"), full_page=True)


def test_benchmark_e_limites(page):
    page.goto(page.base)
    page.click("role=tab[name='Benchmark']")
    page.wait_for_selector("text=não é hardware de borda real")
    page.screenshot(path=str(IMG / "ui_benchmark.png"), full_page=True)
    page.click("role=tab[name='Limites']")
    page.wait_for_selector("text=café brasileiro")
    page.screenshot(path=str(IMG / "ui_limites.png"), full_page=True)
