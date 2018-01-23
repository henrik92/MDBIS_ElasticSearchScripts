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



$json_path = "outputFile.json";
$error = "";

/* JSON Input File lesen */
if (file_exists($json_path)) {
    $json = file_get_contents($json_path);
    $top = json_decode($json, true);
} else {
    $error = "File not found.";
}

/* JSON Items auslesen */
if ($error == "") {
    foreach ((array) $top as $item => $key) {
        $movie_title = $key['title'];
        $rating_value = $key['rating']['value'];
        $rating_counter = $key['rating']['counter'];
        $rating_rank = $key['rating']['rank'];
        $image = $key['info']['image_url'];

        /* Items in neues Array einfügen */
        $params = [
            'index' => 'movies',
            'type' => 'movie',
            'body' => [
                'rating_counter' => $rating_counter,
                'rating_rank' => $rating_rank,
                'rating_value' => $rating_value,
                'title' => $movie_title,
                'image' => $image
              ]
        ];
            $response = $elastic_client->index($params);

            if(!empty($response)){
            echo 'Indexing:' . $params['body']['title'] . ' - Success';
        } else {
            echo 'Failed';
        }
    }
}
