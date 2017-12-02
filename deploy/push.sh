#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR/../

if [ $1 = "prod" ]; then
  SECRETS_BUCKET="numerai-api-ml-secrets"
else
  SECRETS_BUCKET="numerai-api-ml-staging-secrets"
fi

docker build -t 074996771758.dkr.ecr.us-west-2.amazonaws.com/api-ml:$2 --build-arg secrets_bucket=$SECRETS_BUCKET . &&
aws --region us-west-2 ecr get-login --no-include-email | bash &&
docker push 074996771758.dkr.ecr.us-west-2.amazonaws.com/api-ml:$2 &&
./deploy/gen-dockerrun.sh $2
