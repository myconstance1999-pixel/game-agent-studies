"""Pure arithmetic reference check; no Minecraft, model, or author-code execution.

Assume per-mille displacement d=X*dimension/1000 and native camera=d/3.
The inspected low-level engine multiplies camera by 3 then truncates to integer
pixels. Fractions isolate that declared linear conversion from floating-point
error. This does not identify the unavailable GameScaling Minecraft adapter.
"""
from fractions import Fraction
import json
from pathlib import Path


def movements(value: int, dimension: int, steps: int = 4) -> dict:
    """Compare chunk truncation, independent step truncation, and residual carry."""
    desired = Fraction(value * dimension, 1000)
    carry = [int(i * desired / steps) - int((i - 1) * desired / steps)
             for i in range(1, steps + 1)]
    return {"desired": desired, "total": int(desired),
            "naive": steps * int(desired / steps), "carry": carry}


def check() -> dict:
    """Exhaust the declared integer action grid and check endpoint/path bounds."""
    axes = {}
    for axis, dimension in (("X", 640), ("Y", 360)):
        different = max_error = positives = negatives = 0
        for value in range(-1000, 1001):
            row = movements(value, dimension)
            desired, total, naive, carry = (row[k] for k in ("desired", "total", "naive", "carry"))
            assert sum(carry) == total
            assert abs(total - desired) < 1
            for i in range(1, 5):
                assert abs(sum(carry[:i]) - i * desired / 4) < 1
            error = abs(total - naive)
            assert error <= 3
            max_error = max(max_error, error)
            if total != naive:
                different += 1
                positives += int(value > 0)
                negatives += int(value < 0)
            opposite = movements(-value, dimension)
            assert total + opposite["total"] == 0
            assert naive + opposite["naive"] == 0
            assert all(a + b == 0 for a, b in zip(carry, opposite["carry"]))
        example = movements(6, dimension)
        example["desired"] = str(example["desired"])
        axes[axis] = {"dimension": dimension, "actions_checked": 2001,
                      "naive_endpoint_mismatches": different,
                      "positive_mismatches": positives, "negative_mismatches": negatives,
                      "maximum_endpoint_difference_pixels": max_error, "example_value_6": example}
    return {
        "status": "PASS", "evidence_type": "PURE_ARITHMETIC_REFERENCE_CHECK_NOT_GAMEPLAY",
        "assumptions": {"integer_per_mille_range": [-1000, 1000], "native_steps": 4,
                        "nominal_step_ms": 50, "native_low_level_camera_gain": 3,
                        "linear_conversion": "camera=d/3; native pixels=trunc(3*camera)",
                        "arithmetic": "exact rational, not measured Java floating-point behavior",
                        "action_path": "continuous env path; excludes optional VPT agent quantization"},
        "properties": ["carry endpoint equals chunk truncation",
                       "every cumulative carry endpoint is within one pixel of ideal",
                       "naive endpoint discrepancy is at most steps-1 pixels",
                       "positive/negative pairs cancel for total, naive, and every carry step"],
        "axes": axes,
        "limitation": "No observed GameScaling adapter bug or Minecraft/model performance result.",
    }


if __name__ == "__main__":
    result = check()
    path = Path(__file__).with_name("mouse_quantization_result.json")
    with path.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "actions_checked": 4002, "result": str(path)}))
