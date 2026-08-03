"""Pure-data content for the in-app interpretation guide window (see ui.py:_open_guide).

No Tkinter/matplotlib imports here -- this stays in the headless layer, same as interpret.py
and model.py. ui.py is the only consumer, rendering these into a scrollable ttk.Notebook tab
per Guide.

Figures and text are transcribed verbatim from ResFrac's public article "Practical guidelines
for DFIT interpretation using the 'compliance method' procedure from URTeC-2019-123". The PNGs
referenced by GuideFigure.image live in dfit_tool/assets/guide/.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuideFigure:
    image: str      # filename under assets/guide/
    caption: str


@dataclass(frozen=True)
class GuideSection:
    title: str
    body: str                       # paragraphs, joined by "\n\n"
    figures: tuple[GuideFigure, ...] = ()


@dataclass(frozen=True)
class Guide:
    title: str
    intro: str
    sections: tuple[GuideSection, ...]
    source: str


SOURCE = ("Figures and text: ResFrac, \"Practical guidelines for DFIT interpretation\" "
          "(compliance method, URTeC-2019-123).")


POSTCLOSURE_GUIDE = Guide(
    title="Postclosure diagnostics (log-log)",
    intro=(
        "Use the log-log plot to diagnose the postclosure behavior. This interpretation helps you "
        "decide: (a) how to estimate permeability, and (b) how to estimate pore pressure. Note that in "
        "all cases, the derivatives are taken with respect to actual shut-in time, rather than "
        "'superposition time.' As discussed by McClure (2017), the 'superposition time derivative' is "
        "not recommended for DFIT interpretation.\n\n"
        "Slope guide: a -1/2 slope after the derivative peak is postclosure (impulse) linear flow; a "
        "-1 slope is radial. A -1 slope reached immediately after the peak (gas / high-GOR oil) is "
        "'false radial'; a -1 slope is 'genuine radial' only if it develops after a sustained -1/2 "
        "linear period."
    ),
    sections=(
        GuideSection(
            title="PC-A: Postclosure linear flow",
            body=(
                "The figure below shows the 'ideal' DFIT postclosure transient. The log-log derivative plot "
                "peaks and then bends down into a -1/2 slope, corresponding to postclosure linear flow. On the "
                "plot, the red line does not visually appear to have a -1/2 slope, but this is an artifact "
                "because the y-axis is stretched relative to the x-axis.\n\nThe -1/2 slope occurs because "
                "pressure change is scaling with shut-in time to the -1/2 power. A -1 slope would indicate that "
                "pressure change is scaling with shut-in time to the -1 power."
            ),
            figures=(
                GuideFigure(
                    "pc_a.png",
                    "Figure 6: Example of Scenario PC-A: Postclosure linear flow. The dashed black line has a slope of -1/2.",
                ),
            ),
        ),
        GuideSection(
            title="PC-B: False radial",
            body=(
                "False radial is very common in formations where the reservoir fluid is gas or high-GOR "
                "volatile oil. It is characterized by an immediate bend into a -1 slope after the peak in the "
                "derivative. This is not genuine radial flow geometry and should not be used to estimate "
                "permeability. However, it is possible to get a reasonably accurate estimate for pore pressure."
            ),
            figures=(
                GuideFigure(
                    "pc_b.png",
                    "Figure 7: Example of Scenario PC-B: False radial. The green line has a slope of -1.",
                ),
            ),
        ),
        GuideSection(
            title="PC-C: False radial into genuine linear",
            body=(
                "If the transient duration is sufficiently long (or false radial occurs sufficiently early), "
                "then the transient can slip into a -1/2 slope after the false radial signature. This is "
                "uncommon because shut-in usually ends too early for the test to reach genuine linear flow "
                "after false radial. A reminder -- false radial only occurs in gas reservoirs or high GOR "
                "volatile oils.\n\nWith this scenario, the later -1/2 slope can be interpreted as genuine "
                "linear and used to estimate both permeability and pore pressure."
            ),
            figures=(
                GuideFigure(
                    "pc_c.png",
                    "Figure 8: Example of Scenario PC-C: False radial into genuine linear. The green line has a slope of -1. The dashed black line has a slope of -1/2.",
                ),
            ),
        ),
        GuideSection(
            title="PC-D: Genuine linear to genuine radial",
            body=(
                "In the test below, there is an extended -1/2 slope after the peak, followed by a -1 slope. "
                "The -1 slope can be interpreted as genuine radial. This is not common, because in most tests, "
                "the shut-in duration would need to be weeks or months until the test reached genuine radial. "
                "In the test below (which was performed in an oil shale), the permeability was relatively high "
                "for a shale (tens of microdarcy), the injection volume was unusually low (less than 10 bbl), "
                "and the shut-in was unusually long (several weeks). This combination of factors made it "
                "possible to achieve genuine radial prior to the end of the test.\n\nIn this scenario, either "
                "the linear or radial periods can be used to estimate permeability and pore pressure. As a QC, "
                "the estimates can be compared, and they should be similar. In the test shown below, as "
                "expected, the radial and linear permeability estimates were similar."
            ),
            figures=(
                GuideFigure(
                    "pc_d.png",
                    "Figure 9: Example of Scenario PC-D: Genuine linear to genuine radial. The dashed black line has a slope of -1/2. The green line has a slope of -1.",
                ),
            ),
        ),
        GuideSection(
            title="PC-E: Peak reached, trend not established",
            body=(
                "In the test below, the derivative plot reaches a peak but does not progress for sufficient "
                "duration to establish a clear -1/2 slope. With 'adequate' confidence, it is acceptable to "
                "extrapolate the remaining data on a t1/2 trend to estimate pore pressure. This yields an "
                "acceptable, but somewhat uncertain, pore pressure estimate.\n\nThis scenario is fairly common "
                "in practical DFITs, even with the standard one-week shut-in. Larger injection volumes tend to "
                "delay the onset of impulse linear flow, which is one reason why we recommend using relatively "
                "small (10-20 bbl) injection volumes."
            ),
            figures=(
                GuideFigure(
                    "pc_e.png",
                    "Figure 10: Example of Scenario PC-E: Derivative reaches a peak but the postclosure trend is not established.",
                ),
            ),
        ),
        GuideSection(
            title="PC-F: Derivative does not reach a peak",
            body=(
                "In the test below, the derivative curve is still increasing at the end of the test. It is not "
                "possible to estimate pore pressure from this test. Because pore pressure cannot be estimated, "
                "it is also not possible to estimate permeability."
            ),
            figures=(
                GuideFigure(
                    "pc_f.png",
                    "Figure 11: Example of Scenario PC-F: The derivative does not reach a peak.",
                ),
            ),
        ),
        GuideSection(
            title="Pore pressure estimation",
            body=(
                "If the procedure says to 'extrapolate t-1/2', then make a plot of pressure versus shut-in "
                "time to the -1/2 power and draw a straight line through the end of the data to the y-axis "
                "(corresponding to time infinity) (Section 3.1.6 from McClure et al., 2019).\n\nIf the "
                "procedure says to 'extrapolate t-1/2 from peak', then make a plot of pressure versus shut-in "
                "time to the -1/2 power and draw a straight line from the final point in the data to the y-axis "
                "(corresponding to time infinity).\n\nIf the procedure says to 'extrapolate t-1', then make a "
                "plot of pressure versus shut-in time to the -1 power and draw a straight line through the end "
                "of the data to the y-axis. If the procedure says 'none', then it is not possible to estimate "
                "the pore pressure from the test."
            ),
            figures=(
                GuideFigure(
                    "pore_pressure.png",
                    "Figure 12: Example of extrapolating a linear flow period to reciprocal sqrt(t) of zero to estimate pore pressure. Note that this figure is plotting WHP, and so the values would need to be appropriately adjusted for hydrostatic to estimate BHP in the target interval.",
                ),
            ),
        ),
    ),
    source=SOURCE,
)


CLOSURE_GUIDE = Guide(
    title="Closure diagnostics (G-function / dP/dG)",
    intro=(
        "Estimate the magnitude of Shmin and the effective ISIP from the G-function plot. Stress is "
        "estimated from the 'compliance method' procedure based on the dP/dG curve, as described in "
        "Section 3.1.2 of URTeC-2019-123.\n\nThe primary derivative dP/dG usually starts very high and "
        "drops off rapidly. The contact point is picked from the upward deflection in dP/dG after the "
        "initial dropoff. It is necessary to manually adjust the axis scale for dP/dG so that it is "
        "sufficiently low to visualize the shape of the curve after the initial falloff -- the initial "
        "high values are not the region of interest; the 'contact' deflections occur later, at lower "
        "dP/dG values."
    ),
    sections=(
        GuideSection(
            title="C-A: Clear contact point",
            body=(
                "In this scenario, the dP/dG makes a clear \"S\" shape. This is the 'ideal' trend that we see "
                "roughly half of field DFITs, and which is easily reproduced in numerical simulations. The "
                "early-time pressure drop is related to near-wellbore tortuosity. The derivative decreases as "
                "the 'near-wellbore tortuosity' effect dissipates. Then, the derivative increases when the "
                "fracture walls contact because the system becomes stiffer. The 'contact pressure' should be "
                "picked once dP/dG increases roughly 10% from the minimum point. Then, subtract 75 psi to "
                "account for crack roughness at contact, and the result is the 'best estimate' for Shmin.\n\n"
                "You should not use the 'holistic method' concepts of 'tip-extension,' 'pressure-dependent "
                "leakoff,' 'fracture height recession,' or 'transverse storage.' While these things could "
                "hypothetically occur, the G-function plotting techniques used to 'diagnose' these phenomena "
                "are flawed and usually lead to misinterpretation.\n\nThe effective ISIP is estimated by "
                "extrapolating a straight line from the pressure versus G-time plot back to the y-intercept at "
                "G = 0, starting from the point of minimum dP/dG."
            ),
            figures=(
                GuideFigure("c_a.png", "Figure 1: Example of Scenario C-A: Clear contact point"),
            ),
        ),
        GuideSection(
            title="C-B: Adequate contact point",
            body=(
                "In these tests, dP/dG monotonically decreases, instead of showing a clear \"S\" shape with a "
                "min/max. However, stress can still be estimated with 'adequate' confidence if there is an "
                "inflection point in dP/dG -- the point where dP/dG stops curving upwards and starts curving "
                "downward can be used as the contact point. As with the standard pick, the stress estimate "
                "should be the pressure at the contact point minus 75 psi.\n\nFigure 3 shows a more difficult "
                "example, where dP/dG reaches a slight minimum, inflects slightly upwards, then mostly flattens "
                "rather than bending back down. The stress estimate is less confident, but net pressure is "
                "seldom much greater than 500 psi: as long as the stress estimate is within ~500 psi of the "
                "effective ISIP, the pick cannot be too far off."
            ),
            figures=(
                GuideFigure("c_b_1.png", "Figure 2: Example of Scenario C-B: Adequate contact point"),
                GuideFigure("c_b_2.png", "Figure 3: Example of Scenario C-B: Adequate contact point"),
            ),
        ),
        GuideSection(
            title="C-C: No contact point",
            body=(
                "In some tests, we cannot identify a compliance-method contact point and make a stress "
                "estimate. This occurs in tests where not only is dP/dG continuously decreasing, but also, "
                "dP/dG does not have an inflection point -- it is continuously decreasing and continuously "
                "bending upwards. In this test, we must decline to estimate stress.\n\nWithout estimates for "
                "stress and effective ISIP, we are also unable to estimate permeability. However, it is still "
                "possible to estimate pore pressure from these tests."
            ),
            figures=(
                GuideFigure("c_c.png", "Figure 4: Example of Scenario C-C: No contact point"),
            ),
        ),
        GuideSection(
            title="C-D: Rapid closure",
            body=(
                "This is a special case when it is possible to estimate stress, even though dP/dG is "
                "monotonically decreasing and continuously concave up. If you feel it is appropriate to assume "
                "that there is not any near-wellbore tortuosity, then you may interpret these tests as 'rapid "
                "closure.' This could happen, for example, if the well is vertical and so the fracture is "
                "initiating longitudinally along the well.\n\nMonotonic dP/dG occurs because the fracture "
                "closes shortly after shut-in (rapid leakoff into the matrix or preexisting fractures). The "
                "best interpretation is that the fracture is closing rapidly, and the stress estimate is within "
                "several hundred psi of the ISIP.\n\nThe stress estimate is fairly uncertain, so in this case "
                "we cannot estimate the 'net pressure' (ISIP - Shmin). Therefore, even though we have a stress "
                "estimate, we don't have a confident estimate of fracture size and so cannot estimate "
                "permeability. 'Rapid closure' should be considered in the context of injection volume/rate and "
                "permeability -- it is the most common observation in low-rate/low-volume microfrac tests, and "
                "in high-permeability vertical wells."
            ),
            figures=(
                GuideFigure("c_d.png", "Figure 5: Example of Scenario C-D: Rapid closure"),
            ),
        ),
        GuideSection(
            title="Stress estimate procedure",
            body=(
                "If the procedure says to use the 'minimum in dP/dG,' identify the contact pressure once dP/dG "
                "has risen about 10% from the minimum, then subtract 75 psi. To estimate effective ISIP, draw a "
                "straight line through the pressure versus G-time curve (starting from the point of minimum "
                "dP/dG) and pick the y-intercept.\n\nIf the procedure says to use the 'inflection point in "
                "dP/dG', identify the contact pressure shortly after the inflection point of dP/dG (where the "
                "slope stops curving upwards and starts curving downwards), and subtract 75 psi. Estimate "
                "effective ISIP from the y-intercept of the straight line through pressure versus G-time.\n\n"
                "If the procedure says 'within a few 100 psi of the ISIP', estimate the literal ISIP (pressure "
                "versus G-time, at the deviation from the straight line), then subtract 100-250 psi as an "
                "approximate range for Shmin.\n\nIf the procedure says 'none', then it is not possible to "
                "estimate the minimum principal stress from the test."
            ),
        ),
    ),
    source=SOURCE,
)
