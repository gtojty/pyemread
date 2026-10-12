# License of the example data

The example eye-movement recordings, audio recordings and derived example
files in `examples/oralReading/`, `examples/fixation_table/` and
`examples/bids_example/` were collected by the author at Haskins
Laboratories for Gong and Shuai (2023), with an SR Research EyeLink tracker
at 250 Hz. The author releases them under the Creative Commons Attribution
4.0 International License (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/). Please cite Gong and Shuai
(2023) when you use them.

The data were collected under a protocol approved by the Yale University
Institutional Review Board, and participants gave informed consent, which
covered public sharing. Participants are identified only by numeric codes;
session dates and file paths were removed from the ASC files. The audio
recordings contain the participants' voices; their public sharing was
covered by the consent.

The passages in `examples/textRasters/` were written for Gong and Shuai
(2023); they and the bitmaps and region files rendered from them are
licensed under CC BY 4.0 as well.

## Fonts for Chinese, Japanese and Korean

The Chinese, Japanese and Korean examples in `examples/textRasters/` were
rendered with the fonts SimSun (`simsun.ttc`; Chinese and Japanese) and
Batang (`batang.ttc`; Korean). These fonts are not distributed with
pyemread. They are included with Microsoft Windows (Batang through the
optional Korean supplemental fonts), in `C:\Windows\Fonts\simsun.ttc` and
`C:\Windows\Fonts\batang.ttc`, and can be passed to `gen.praster` from there. On other systems, use a font
whose license permits it, for example Noto Sans CJK or Noto Serif CJK
(https://github.com/notofonts/noto-cjk, SIL Open Font License). Bitmaps and
region files rendered with another font differ in their coordinates from
the distributed examples. The tests that need these fonts look for them in
the folder named by the environment variable `PYEMREAD_CJK_FONTS`, in
`C:\Windows\Fonts`, and in `examples/textRasters/`; otherwise they use Noto
Sans CJK if it is installed, and they are skipped if no CJK font is found.

Gong, T., & Shuai, L. (2023). Segmented relations between online reading
behaviors, text properties, and reader–text interactions: An eye-movement
experiment. *Frontiers in Psychology, 13*, 1006662.
https://doi.org/10.3389/fpsyg.2022.1006662
