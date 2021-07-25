dvc init

dvc remote add s3remote -d s3://20210723mlops
dvc remote modify s3remote profile hui
dvc remote modify s3remote credentialpath /Users/huiren/.aws/credentials
dvc remote modify s3remote region us-west-2

dvc add data
dvc push --remote s3remote

dvc run -n prepare -d fake_data.csv -d prepare.py -o X.csv -o y.csv python ./prepare.py
dvc run -n train -d X.csv -d y.csv -d train.py -p C python ./train.py

dvc run -n evaluate \
          -d validate.py -d model.pkl \
          -M validation.json \
          python validate.py model.pkl validation.json

dvc exp run --set-param param=100
dvc exp show

dvc run -n pipeline \
        -p n_estimators
        -d train_model.py-d winquality-red.csv \
        -M f1.json \
        python ./train_model.py

dvc metrics show
dvc exp run --set-param n_estimators=100
dvc exp diff
dvc exp show

uvicorn main:app --reload

heroku create demo-app-20210716 --buildpack heroku/python
heroku buildpacks --app demo-app-20210716
git init
git add
git commmit
heroku git:remote --app demo-app-20210716
git push heroku master
heroku run bash --app demo-app-20210716