from flair.embeddings import WordEmbeddings, DocumentRNNEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence

# instantiate pre-trained word embeddings
word_embeddings = WordEmbeddings("glove")

# document pool embeddings - try to exchange this with DocumentRNNEmbeddings!
document_embeddings = DocumentPoolEmbeddings( \
	[word_embeddings], fine_tune_mode="none")

# create an example sentence object
sentence = Sentence("Colorless green ideas sleep furiously.")

# embed the sentence with the document embeddings (needed for each movie review)
document_embeddings.embed(sentence)

# check out the embedded sentence - it's a torch.Tensor object :-)
print(sentence.get_embedding())