# License of the example data

The example eye-movement recordings, audio recordings and derived example
files in `examples/oralReading/`, `examples/fixation_table/` and
`examples/bids_example/` are from Gong and Shuai (2023). They are licensed
under the Creative Commons Attribution 4.0 International License (CC BY 4.0,
https://creativecommons.org/licenses/by/4.0/). Please cite Gong and Shuai
(2023) when you use them.

The data were collected under a protocol approved by the Yale University
Institutional Review Board. Participants gave informed consent, which
covered the public sharing of de-identified data, including the audio
recordings of their reading. Participants are identified only by numeric
codes.

The passages in `examples/textRasters/` were written for Gong and Shuai
(2023); they and the bitmaps and region files rendered from them are
licensed under CC BY 4.0 as well.

## Fonts for Chinese, Japanese and Korean

The Chinese, Japanese and Korean examples in `examples/textRasters/` were
rendered with the fonts SimSun (`simsun.ttc`; Chinese and Japanese) and
Batang (`batang.ttc`; Korean). These fonts are not distributed with pyemread
because their licenses do not allow it. They are installed with Microsoft
Windows (`C:\Windows\Fonts\simsun.ttc` and `C:\Windows\Fonts\batang.ttc`)
and can be passed to `gen.praster` from there. On other systems, use a font
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
