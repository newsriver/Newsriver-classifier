



JOB_NAME=category_classifier_$(date -u +%y_%m_%d_%H_%M_%S)
BUCKET=newsriver-category-classifier
OUTPUT_PATH=gs://$BUCKET/$JOB_NAME

if [ 1 -eq 1 ]; then

gsutil cp projector_config.pbtxt  gs://newsriver-category-classifier/$JOB_NAME/

gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.4 \
--module-name trainer.task \
--package-path trainer/ \
--region europe-west1 \
--scale-tier BASIC \
--config config.yaml \
-- \
--dataPath=gs://$BUCKET/data \
--output_dir=$OUTPUT_PATH \
--eval_steps 100 \
--train_steps 5000
fi


#--scale-tier BASIC        $0.2774 / h
#--scale-tier BASIC_GPU    $1.2118 / h
#--scale-tier STANDARD_1   $2.9025 / h

if [ 0 -eq 1 ]; then
rm -R outputdir/
mkdir outputdir
gcloud ml-engine local train \
    --job-dir /Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/outputdir \
    --module-name trainer.task \
    --package-path /Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/trainer/ \
    --verbosity debug \
    -- \
    --dataPath=gs://$BUCKET/data \
    --output_dir=/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/outputdir \
    --eval_steps 10 \
    --train_steps 500
fi


#python -m trainer.task  --dataPath=gs://newsriver-category-classifier/data --output_dir=gs://newsriver-category-classifier/catClassV1 --train_steps=500 --eval_steps=50
