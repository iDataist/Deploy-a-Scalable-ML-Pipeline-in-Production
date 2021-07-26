# Deploying a Scalable ML Pipeline in Production

## Overview
I developed a CI/CD pipeline to predict salary range based on publicly available Census Bureau data. I created tests to monitor the machine learning pipeline. Then, I deployed the model using the FastAPI package and create API tests. The tests were incorporated into the CI/CD framework using GitHub Actions.

## Dependencies
* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

* Initialize Git and DVC.
   * Continually commit changes. Trained models can be committed to DVC.
    * Connect your local Git repository to GitHub.

* Set up S3

    * Install the<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html" target="_blank"> AWS CLI tool</a>.

    * Create an IAM user with the appropriate permissions. The full instructions can be found <a href="https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console" target="_blank">here</a>, what follows is a paraphrasing:

        * Sign in to the IAM console <a href="https://console.aws.amazon.com/iam/" target="_blank">here</a> or from the Services drop down on the upper navigation bar.
        * In the left navigation bar select **Users**, then choose **Add user**.
        * Give the user a name and select **Programmatic access**.
        * In the permissions selector, search for S3 and give it **AmazonS3FullAccess**.
        * Tags are optional and can be skipped.
        * After reviewing your choices, click create user.
        * Configure your AWS CLI to use the Access key ID and Secret Access key.

* GitHub Actions
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.
    * Add your <a href="https://github.com/marketplace/actions/configure-aws-credentials-action-for-github-actions" target="_blank">AWS credentials to the Action</a>.
    * Set up <a href="https://github.com/iterative/setup-dvc" target="_blank">DVC in the action</a> and specify a command to `dvc pull`.

* API Deployment
    * Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
    * Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Set up access to AWS on Heroku, if using the CLI: `heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy`
