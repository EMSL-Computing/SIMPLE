# Trajectory analyses and metal oxygen coordination

Run `python analyses.py`, then choose **Additional analyses**, or run
`python analyses.py --trajectory` to open this menu directly.
The menu lists RMSD, RMSF, radius of gyration, RDF, distance, and metal O-CN /
metal–O distances. Enter individual numbers, ranges such as `1,3-5`, or **0 / A**
to request all listed analyses. Required atom masks are still requested; cases
without detected metals skip the metal analysis.

For metal coordination, select one or several listed metals (or 0 / A for all),
separately for each input case. The default oxygen-neighbor cutoff is **4 Å** and
is adjustable. This is a geometric neighbor count, not a bond-order assignment
or a universal chemical CN definition. Use the same deliberate cutoff when
comparing systems. This analysis cutoff is independent of FreeE's 2.5 Å preview.

Choose a frame stride and one of:

- Whole trajectory.
- Last N ns (default N = 10).
- Last N frames.
- Specific start/end time in ns (saved times within the interval, inclusive).

Saved trajectory timestamps are used when available. Amber ASCII mdcrd contains
no timestamps: enter the saved-frame interval in ps (`MD dt * ntwx`); its analysis
time axis starts at zero. This interval is not the MD integration timestep alone.
Time windows require increasing timestamps, and an empty selected interval is
reported as an error. The same selected frames feed CN, distance statistics and
plots. With stride > 1, contacts between sampled frames are not observed.

## Outputs

Each case has CSV and TSV data, PNG plots, and text/JSON summaries. For a metal
at topology atom 123:

- `metal_123_oxygen_cn.tsv`: time (ns), zero-based frame index, total O-CN,
  non-water O-CN and water O-CN. The PNG shows the temporal profile.
- `metal_123_oxygen_cn_statistics.tsv`: mean, population SD, minimum and maximum
  for all three CN counts over the analyzed frames.
- `metal_123_oxygen_distances.tsv`: distance for each tracked O at every selected
  frame, whether it is inside the cutoff, and chain/segment, residue number/name,
  atom name and one-based topology atom index, e.g. `A:4ASP@OD1 [#123]`.
- `metal_123_oxygen_distances_statistics.tsv`: mean/SD/min/max distance over
  **all selected frames**, separately reported contact-only distance statistics,
  and contact occupancy (fraction of selected frames inside the cutoff).
- `metal_123_oxygen_distances.png`: temporal distance curves with a cutoff line.
  Additional `_page_2.png`, etc. show remaining contacts, 12 per page.
- `combined_summary.tsv`: case/analysis summary; CSV copies remain available.

Tracked O identities are the union of all O atoms that enter the cutoff in at
least one **analyzed** frame, not just those present in the first snapshot.
CN is recomputed dynamically from every O, including water. If the shell is
empty throughout, CN is zero and no artificial distance series is generated.
The box's minimum-image distance is used when valid periodic box data exist;
otherwise distances are Cartesian. The summary records this and the number of
periodic frames. Standard deviation is **population SD (ddof = 0), not SEM** or
a free-energy uncertainty. RMSD/RMSF alignment uses independent copies, so it
does not rotate/alter coordinates used for coordination and other analyses.

No TI integration, endpoint selection, or restraint free-energy correction is
performed by this trajectory analysis.
