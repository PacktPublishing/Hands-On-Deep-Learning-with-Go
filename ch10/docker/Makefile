cpu-image:
		mkdir -p staging/
		cp model_wrapper.sh staging/
	    docker build --no-cache -t "ACCOUNTID.dkr.ecr.ap-southeast-2.amazonaws.com/$(MODEL_CONTAINER):$(VERSION_TAG)" .
		rm -rf staging/

cpu-push: cpu-image
	    docker push "ACCOUNTID.dkr.ecr.ap-southeast-2.amazonaws.com/$(MODEL_CONTAINER):$(VERSION_TAG)"
