#!/bin/bash

# Usage: ./deploy-new-environment config-name environment-name
#
# Example: ./deploy-new-environment api-ml-production api-ml-production 

eb config put $1
eb create --cfg $1 $2
