# Example data

| Folder / file | Contents |
|---|---|
| `oralReading/` | EyeLink 1000 recordings (`<id>/<id>.asc`, binocular, 250 Hz) of three participants (1950138, 1950149, 1950168) reading three English passages aloud, the audio recorded during each trial (`<id>/<id>-<trial>.wav`), example animations (`.mp4`), and the region files of the passages (`story01–03.region.csv`). From Gong and Shuai (2023), redistributed with permission. |
| `textRasters/` | Passages (`.txt`), bitmaps and region files in English, Chinese, Japanese and Korean, and the CJK fonts used to render them. |
| `bids_example/` | Eye-Tracking-BIDS version of the right-eye data of participant 1950138 (created by `make_bids_example.py`). |
| `example_workflow.py` | The complete tutorial of the accompanying article. |
| `tutorial.ipynb` | The same tutorial as a notebook (runs on Google Colab). |

Outputs of the tutorial are written to `tutorial_output/` and are not stored in the repository.

Gong, T., & Shuai, L. (2023). Segmented relations between online reading behaviors, text properties, and reader–text interactions: An eye-movement experiment. *Frontiers in Psychology, 13*, 1006662. https://doi.org/10.3389/fpsyg.2022.1006662
