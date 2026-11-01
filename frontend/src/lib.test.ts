import { describe, expect, it } from "vitest";
import { csvEscape, domainTag, pct, prPath, scaleBox } from "./lib";

describe("lib", () => {
  it("scaleBox escala proporcionalmente", () => {
    expect(scaleBox([100, 50, 200, 150], 1000, 500, 500, 250)).toEqual([50, 25, 100, 75]);
  });
  it("pct trata nulos e NaN", () => {
    expect(pct(null)).toBe("—");
    expect(pct(Number.NaN)).toBe("—");
    expect(pct(0.8214)).toBe("82.1%");
  });
  it("domainTag marca milho como outro domínio", () => {
    expect(domainTag("maize_test")).toContain("zero-shot");
    expect(domainTag("sudars_weed")).toBe("mesmo domínio");
    expect(domainTag("sudars_weed", "test_origin")).toBe("origem retida");
  });
  it("csvEscape protege vírgulas e aspas", () => {
    expect(csvEscape('a,"b"')).toBe('"a,""b"""');
    expect(csvEscape("weed")).toBe("weed");
  });
  it("prPath gera M/L", () => {
    expect(prPath([0, 1], [1, 0.5], 100, 100)).toBe("M0.0,0.0 L100.0,50.0");
  });
});
