dvc init
dvc remote add s3remote -d s3://20210723mlops
dvc remote modify s3remote profile hui
dvc remote modify s3remote credentialpath /Users/huiren/.aws/credentials
dvc remote modify s3remote region us-west-2
dvc add data
dvc push --remote s3remote

uvicorn main:app --reload

heroku create app-20210725 --buildpack heroku/python
heroku apps
# install a buildpack that allows the installation of apt-files
# define the Aptfile that contains a path to DVC
heroku buildpacks:add --index 1 heroku-community/apt
heroku buildpacks --app app-20210725
heroku git:remote --app app-20210725
git push heroku HEAD:master
heroku run bash --app app-20210725