from natasha import (
    Segmenter,
    MorphVocab,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,

    PER,
    NamesExtractor,
    DatesExtractor,
    MoneyExtractor,
    AddrExtractor,

    Doc
)

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
money_extractor = MoneyExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)


text = '''# -*- coding: utf-8 -*-\nПроизводим БОР 11% жидкое удобрение АГРОМАКС 150 гр/литр подкормка - Отгрузки по РФ; цена: 295 руб / л.; Мы производитель. Отгрузки по РФ. Агромакс Бор 11% - высококонцентрированное борное удобрение подкормка, позволяющее компенсировать нехватку бора в растениях в течение всей вегетации путем внекорневых подкормок. Содержит бор в виде хелата и мембранный проникатель в составе, благодаря чему обеспечивается полное поглощение бора растением и увеличивает подвижность элемента внутри растительных тканей. Бор обеспечивает стойкость точки роста к неблагоприятным погодным условиям, предотвращая нарушение ростовых процессов и поддерживая баланс фитогормонов в растении.

Состав:

Бор 11 % (В 150 г/л, N - 60 г/л)"
2;2020-10-27'''

doc = Doc(text)

markup = ner_tagger(text)
print(markup.spans)
#print(markup.print())

doc.segment(segmenter)
print(doc.tokens[:5])
print(doc.sents[:5])


doc.parse_syntax(syntax_parser)
print(doc.tokens[:5])
doc.sents[0].syntax.print()
