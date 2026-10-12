# Example data

| Folder / file | Contents |
|---|---|
| `oralReading/` | SR Research EyeLink recordings (`<id>/<id>.asc`, binocular, 250 Hz) of three participants (1950138, 1950149, 1950168) reading three English passages aloud, the audio recorded during each trial (`<id>/<id>-<trial>.wav`), and the region files of the passages (`story01–03.region.csv`). Collected for Gong and Shuai (2023); licensed under CC BY 4.0. |
| `textRasters/` | Passages (`.txt`), bitmaps and region files in English, Chinese, Japanese and Korean. The Chinese, Japanese and Korean examples were rendered with the Windows fonts SimSun and Batang, which are not distributed; see LICENSE-DATA.md for where to find them and for free alternatives. |
| `bids_example/` | Eye-Tracking-BIDS version of the right-eye data of participant 1950138 (created by `make_bids_example.py`). |
| `fixation_table/` | The right-eye fixations of participant 1950138 as a plain CSV file of the kind exported by other software (created by `make_fixation_table_example.py`). |
| `example_workflow.py` | The complete tutorial of the accompanying article. |
| `tutorial.ipynb` | The same tutorial as a notebook (runs on Google Colab). |

Outputs of the tutorial are written to `tutorial_output/` and are not stored in the repository.

The example data are licensed under CC BY 4.0; see [LICENSE-DATA.md](LICENSE-DATA.md), which also gives the ethics approval and consent.

Gong, T., & Shuai, L. (2023). Segmented relations between online reading behaviors, text properties, and reader–text interactions: An eye-movement experiment. *Frontiers in Psychology, 13*, 1006662. https://doi.org/10.3389/fpsyg.2022.1006662
