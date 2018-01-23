<?php
require 'vendor/autoload.php';


//Initializing Elastic Client
use Elasticsearch\ClientBuilder;

$hosts = [
    'http://search-mdbis-x25ypnphotzm5cnlfacv4nw7xq.eu-central-1.es.amazonaws.com:80'
  ];

$clientBuilder = ClientBuilder::create();
$clientBuilder->setHosts($hosts);
$elastic_client = $clientBuilder->build();

//Create Index
$params = [
    'index' => 'movies'
  ];

$response = $elastic_client->delete($params);
echo $response;
        ?>
