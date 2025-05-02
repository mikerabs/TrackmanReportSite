# Trackman webapp commands

If you haven't loaded this at all yet, start the Docker engine on your computer and run:

docker-compose -f compose-dev.yaml up --build

This will build everything and load all the libraries from requirements.txt 


Once you've built it, remove the --build flag when running it:

docker-compose -f compose-dev.yaml up
