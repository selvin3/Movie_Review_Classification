import argparse
import os
from model import build_model

import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers



def text_to_vetor(vectorizer, dataframe):
    """Return text vector and labels"""
    text_set = dataframe['Phrase'].values
    labels = dataframe['Sentiment'].values
    vectorizer.adapt(text_set)
    train_text = []
    for text in tqdm(text_set):
        train_text.append(vectorizer(text))
    return train_text, labels


def text_dataset(file_path, vocab_size, batch_size):
    """Return dataset"""
    df = pd.read_csv(file_path, sep = '\t')
    df["Phrase"] = df['Phrase'].replace(r'[^\w\s]', '',regex=True)
    df['Phrase'] = df['Phrase'].replace(r'^\s*$', '', regex=True)
    df = df[df['Phrase'] != '']
    maxlen = max([len(x) for x in df['Phrase']])

    # define text vectorizer
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size, 
        standardize='lower_and_strip_punctuation',
        output_mode='int',
        output_sequence_length=maxlen,
        pad_to_max_tokens=True,
    )

    train_vec, labels = text_to_vetor(vectorizer=vectorizer, dataframe=df)
    dataset =  tf.data.Dataset.from_tensor_slices((train_vec, labels))
    dataset = dataset.shuffle(2048, seed = 42)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset.batch(batch_size), maxlen
   
def train_test_split(dataset):
    """Return train and validation dataset"""
    validation_size = int(0.2 * len(dataset))
    train_size = int(0.8*len(dataset))
    train_dataset = dataset.take(train_size)
    validation_dataset = dataset.skip(train_size).take(validation_size)
    return train_dataset, validation_dataset


def train(file_path, checkpoint_path, batch_size, epochs, embed_dim, num_heads,ff_dim, vocab_size, num_class):
    # load dataset
    dataset, maxlen = text_dataset(file_path = file_path, vocab_size=vocab_size ,batch_size = batch_size)
    train_ds, validation_ds = train_test_split(dataset=dataset)


    #define model
    transformer_model = build_model(embed_dim=embed_dim, num_heads=num_heads,ff_dim=ff_dim, maxlen = maxlen, vocab_size=vocab_size, num_class=num_class)
    inputs = layers.Input(shape=(maxlen,), batch_size=batch_size)
    outputs = transformer_model(inputs)
    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    best_checkpoint_path = os.path.join(checkpoint_path,'best_ckpt')
    every_checkpoint_path =  os.path.join(checkpoint_path, 'ckpt')

    # Save best model checkpoint.
    best_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_checkpoint_path, save_best_only=True, monitor = 'val_accuracy')

    # Save every model checkpoint.
    every_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=every_checkpoint_path, save_freq="epoch")
    
    # If the validation loss fails to improve during training, the model training process is halted to prevent overfitting.
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience = 5,
        restore_best_weights = True,
    )

    model.compile(
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics = ['accuracy']
    )



    model.fit(train_ds, validation_data=validation_ds ,epochs = epochs, verbose=1, callbacks = [best_checkpoint_cb, every_checkpoint_cb, early_stopping_cb])
    

    return 0

def main():
    parser = argparse.ArgumentParser()
    ## required parameters
    parser.add_argument("--mode", default=None, required=True, type=str, help="train/test mode")
    parser.add_argument("--data_file", default=None, type=str, required=True, help="The input train/test data file")
    parser.add_argument("--checkpoint_dir", default=None, type = str, required= True, help="File path to save checkpoints at model training time")
    parser.add_argument("--output_dir", default=None, type=str, help="File path to save submission file at model test time")
    parser.add_argument("--batch_size", default=64 , type=int, help="Batch size")
    parser.add_argument("--embed_dim", default=256 , type=int, help="Embedding size for each token")
    parser.add_argument("--num_heads", default=8, type=int, help="Number of attention heads")
    parser.add_argument("--ff_dim", default=256, type=int, help="Hidden layer size in feed forward network inside transformer")
    parser.add_argument("--vocab_size", default=30000, type=int, help="Size of vocabulary")
    parser.add_argument("--num_class", default=5, type=int, help="Number of output class to classify")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number epochs for train a model")

    args = parser.parse_args()

    if args.mode == 'train':
        train(file_path=args.data_file, checkpoint_path=args.checkpoint_dir, batch_size=args.batch_size, epochs=args.num_epochs, embed_dim= args.embed_dim, num_heads=args.num_heads, ff_dim=args.ff_dim,vocab_size=args.vocab_size, num_class=args.num_class)
        
    
if __name__ == '__main__':
    main()