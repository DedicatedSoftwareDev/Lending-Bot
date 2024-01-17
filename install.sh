#https://www.arangodb.com/download-major/ubuntu/
#Docker sudo docker run -p 8529:8529 -e ARANGO_ROOT_PASSWORD=openSesame arangodb/arangodb:3.7.12
sudo apt install curl -y
curl -OL https://download.arangodb.com/arangodb37/DEBIAN/Release.key
sudo apt-key add - < Release.key
echo 'deb https://download.arangodb.com/arangodb37/DEBIAN/ /' | sudo tee /etc/apt/sources.list.d/arangodb.list
sudo apt-get install apt-transport-https -y
sudo apt-get update
sudo apt-get install arangodb3=3.7.12-1
git submodule init
git submodule update
sudo apt install python3-pip
pip install --upgrade poloniexapi pyArango pandas sklearn keras tensorflow