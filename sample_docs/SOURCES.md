# Sample corpus: sources and licensing

Five public-domain texts in three languages, so that anyone can run the app and the benchmark
without the private corpus. Machine-readable provenance, page counts and SHA-256 hashes are in
[`corpus.json`](corpus.json); `tests/test_corpus.py` checks them.

| File | Language | Work | Upstream source | Why it may be redistributed |
|---|---|---|---|---|
| `tr_teskilati_esasiye_1924.pdf` | Turkish | Teşkîlât-ı Esâsîye Kanunu (Constitution of Turkey, 1924), amended text | [tr.wikisource](https://tr.wikisource.org/wiki/Te%C5%9Fk%C3%AEl%C3%A2t-%C4%B1_Es%C3%A2s%C3%AEye_Kanunu_(1924)) | Officially published legislation; Turkish copyright law (FSEK) makes its reproduction free |
| `tr_omer_seyfettin_hikayeler.pdf` | Turkish | Ömer Seyfettin, "Kaşağı" and "Başını Vermeyen Şehit" | [tr.wikisource](https://tr.wikisource.org/wiki/Ka%C5%9Fa%C4%9F%C4%B1) | Author died 1920 (life + 70 years) |
| `en_us_constitution.pdf` | English | U.S. Constitution, 1787 text without amendments | [Project Gutenberg #5](https://www.gutenberg.org/ebooks/5) | Public-domain U.S. founding document |
| `en_scandal_in_bohemia.pdf` | English | Arthur Conan Doyle, "A Scandal in Bohemia" (1891) | [Project Gutenberg #1661](https://www.gutenberg.org/ebooks/1661) | Author died 1930 (life + 70 years) |
| `de_kafka_die_verwandlung.pdf` | German | Franz Kafka, "Die Verwandlung" (1915) | [Project Gutenberg #22367](https://www.gutenberg.org/ebooks/22367) | Author died 1924 (life + 70 years) |

## How the files are made

- `source/<name>.txt` is the canonical text: the first line is the title, paragraphs are separated
  by blank lines.
- `python scripts/build_sample_docs.py` renders each text to an A4 PDF with the bundled DejaVu Sans
  font and a `- N -` footer on every page, so page citations can be checked by eye. The output is
  byte-identical on every platform with the pinned `fpdf2`; `--check` verifies this.
- The texts were downloaded from the upstream sources above for v1 of the corpus (git commit
  `5770964`). Only the PDFs were committed then, so the texts were recovered from those PDFs, word
  for word, by `scripts/legacy/recover_sources_from_v1_pdfs.py`. That step also removed HTML
  entities (`&mdash;`, `&acirc;`, …) and wiki indentation colons that v1 had left in the Ömer
  Seyfettin text, and the story title that v1 printed twice.
- The Project Gutenberg header and footer were stripped, as Project Gutenberg asks when only the
  public-domain text is redistributed.

## Notes on the texts

- The 1924 constitution is the version published on Turkish Wikisource. It contains amended
  articles: for example, Articles 10–11 give women the right to vote and to be elected, which was
  added by the 1934 amendment. It is therefore not the original 1924 wording. Which amendment date it
  corresponds to as a whole has not been verified.
- Upstream typos are kept as they are, for example "teklif dehi edilemez" in Article 102.
