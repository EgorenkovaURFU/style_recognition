pipeline {
    agent any
    environment {
        JENKINS_HOME = "$JENKINS_HOME"
        BUILD = "${JENKINS_HOME}/workspace/style_recognition"
        DOCKER_IMAGE_NAME = 'style_recognition'
        DOCKER_IMAGE_TEST = 'test_application'
    }

    stages{

        stage( 'Installation of modules'){
            steps{
                sh 'pip install pytest'
                sh 'pip install streamlit'
                sh 'pip install dvc'
                sh 'pip install dvc-gdrive'
            }
        }


        // stage('Build Doker image to test application') {
        //     sh 'docker build -f test.Dockerfile -t ${DOCKER_IMAGE_TEST}'
        // }

        // stage('RUN Tests') {
        //     steps{
        //         sh 'docker run -rm ${DOCKER_IMAGE_TEST}'
        //     }
        // }

        stage('Build Docker image'){
            steps {
                sh 'docker build -f app.Dockerfile -t ${DOCKER_IMAGE_NAME} .'
            }
        }
        
        stage( 'RUN Docker'){
            steps{
                sh 'docker run --rm -d -v $(pwd)/result:/app/result -p 8501:8501 --name style_recognition-app ${DOCKER_IMAGE_NAME}'
                // sh 'docker exec style_recognition-app bash'
                // withEnv(["HOME=${env.WORKSPACE}"]) {
                //     sh 'pytest ./test/test_main.py'}
                // sh 'exit'
            }

        }
        

        // stage( 'RUN Test'){
        //     steps{
        //         sh 'python3 test/test_main.py'
        //     }
        // }
        stage( 'DVC'){
            steps{

                sh 'dvc add result'
                sh 'dvc commit -m "результат"'
                sh 'dvc push '
            }
        }
    }
}
