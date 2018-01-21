PROBLEM=text2fluff
DATA_DIR=/app/sandbox/cresta/ai/sandbox/tim/data/parallel
TMP_DIR=/tmp/t2t_text2fluff
TRAIN_DIR=/tmp/t2t_text2fluff_train
MODEL=transformer
mkdir -p $DATA_DIR $TMP_DIR

if [ $1 = data ]; then
    t2t-datagen \
    --data_dir=$DATA_DIR \
    --tmp_dir=$TMP_DIR \
    --problem=$PROBLEM
fi

if [ $1 = train ]; then
    t2t-trainer \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=text2fluff_hparams \
    --output_dir=$TRAIN_DIR \
    --worker_gpu=3
fi

if [ $1 = interactive ]; then
    t2t-decoder \
    --data_dir=$DATA_DIR \
    --problems=$PROBLEM \
    --model=$MODEL \
    --hparams_set=text2fluff_hparams \
    --output_dir=$TRAIN_DIR \
    --decode_interactive
fi

if [ $1 = clean ]; then
    rm -rf $TRAIN_DIR
    rm -rf $TMP_DIR
    rm $DATA_DIR/text2fluff-*
fi
