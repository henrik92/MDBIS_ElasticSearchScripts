<?php

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

require 'vendor/autoload.php';


//Initializing Elastic Client
use Elasticsearch\ClientBuilder;

$hosts = [
    'http://search-mdbis-x25ypnphotzm5cnlfacv4nw7xq.eu-central-1.es.amazonaws.com'
];

$clientBuilder = ClientBuilder::create();
$clientBuilder->setHosts($hosts);
$elastic_client = $clientBuilder->build();

//Create Index
$params = [
    'index' => 'movies',
    'body' => [
        'settings' => [
            'number_of_shards' => 1,
            'number_of_replicas' => 1
        ],
        'mappings' => [
            'movie' => [
                '_source' => [
                    'enabled'=> true
                ],
                'properties' => [
                    'title' => [
                        'type' => 'text',
                        'analyzer' => 'standard'
                    ],
                    'rating_value' => [
                        'type' => 'float',
                    ],
                    'rating_counter' => [
                        'type' => 'integer',
                    ],
                     'rating_rank' => [
                        'type' => 'integer',
                    ],
                ]
            ]
        ]
    ]
    ];
$elastic_client->indices()->create($params);
echo 'Success';
?>
//Create Mapping
