import logging
import os

import datasets as ds

logger = logging.getLogger(__name__)

_CITATION = """\
- 吉越 卓見, 河原 大輔, 黒橋 禎夫: 機械翻訳を用いた自然言語推論データセットの多言語化, 第244回自然言語処理研究会, (2020.7.3).
- Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing (EMNLP).
- Peter Young, Alice Lai, Micah Hodosh, and Julia Hockenmaier. "From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions." Transactions of the Association for Computational Linguistics 2 (2014): 67-78.
"""

_DESCRIPTION = """\
== 日本語SNLI(JSNLI)データセット ==

SNLI コーパスを日本語に翻訳した自然言語推論データセット
学習データは元データを翻訳し、計算機によるフィルタリングによって作成
評価データは日本語として意味が通るか、翻訳後のラベルが元のラベルと一致しているかどうかの2段階のクラウドソーシングによりデータをフィルタリング
"""

_HOMEPAGE = "https://nlp.ist.i.kyoto-u.ac.jp/?%E6%97%A5%E6%9C%AC%E8%AA%9ESNLI%28JSNLI%29%E3%83%87%E3%83%BC%E3%82%BF%E3%82%BB%E3%83%83%E3%83%88"

_LICENSE = """\
CC BY-SA 4.0
"""

_URL = "https://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=https://nlp.ist.i.kyoto-u.ac.jp/nl-resource/JSNLI/jsnli_1.1.zip&name=JSNLI.zip"


class JSNLIDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.1.0")  # type: ignore

    BUILDER_CONFIGS = [
        ds.BuilderConfig(
            name="with-filtering",
            version=VERSION,  # type: ignore
            description="SNLIの学習データに機械翻訳を適用した後、BLEUスコアの閾値0.1でフィルタリングを施したもの。BERTにこの学習データを学習させることにより、93.0%の精度を記録した。(533,005ペア)",
        ),
        ds.BuilderConfig(
            name="without-filtering",
            version=VERSION,  # type: ignore
            description="SNLIの学習データに機械翻訳を適用したもの。フィルタリングは行っていない。(548,014ペア)",
        ),
    ]

    def _info(self) -> ds.DatasetInfo:
        features = ds.Features(
            {
                "premise": ds.Value("string"),
                "hypothesis": ds.Value("string"),
                "label": ds.ClassLabel(
                    names=["entailment", "neutral", "contradiction"]
                ),
            }
        )
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        jsnli_base_dir = dl_manager.download_and_extract(_URL)
        jsnli_dir = os.path.join(
            jsnli_base_dir, f"jsnli_{self.VERSION.major}.{self.VERSION.minor}"  # type: ignore
        )

        train_w_filtering_path = os.path.join(jsnli_dir, "train_w_filtering.tsv")
        train_wo_filtering_path = os.path.join(jsnli_dir, "train_wo_filtering.tsv")

        dev_path = os.path.join(jsnli_dir, "dev.tsv")
        if "with-filtering" in self.config.name:
            tng_path = train_w_filtering_path
        elif "without-filtering" in self.config.name:
            tng_path = train_wo_filtering_path
        else:
            raise ValueError(f"Invalid config name: {self.config.name}")

        tng_gen_kwargs = {
            "tsv_path": tng_path,
        }
        val_gen_kwargs = {
            "tsv_path": dev_path,
        }

        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,  # type: ignore
                gen_kwargs=tng_gen_kwargs,  # type: ignore
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs=val_gen_kwargs,  # type: ignore
            ),
        ]

    def _generate_examples(  # type: ignore
        self,
        tsv_path: str,
    ):
        with open(tsv_path, "r") as rf:
            for sentence_id, line in enumerate(rf):
                label, premise, hypothesis = line.replace("\n", "").split("\t")

                example_dict = {
                    "label": label,
                    "premise": premise,
                    "hypothesis": hypothesis,
                }
                yield sentence_id, example_dict
