from tclogger import logger
from elasticsearch import Elasticsearch

from configs.envs import ELASTIC_ENVS


class ElasticSearchClient:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def connect(self):
        """Connect to self-managed cluster with API Key authentication
        * https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html#auth-apikey

        How to create API Key:
        - Go to Kibana: http://<hostname>:5601/app/management/security/api_keys
        - Create API Key, which would generated a json with keys "name", "api_key" and "encoded"
        - Use "encoded" value for the `api_key` param in Elasticsearch class below

        Connect to self-managed cluster with HTTP Bearer authentication
        * https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html#auth-bearer
        """
        if self.verbose:
            logger.note("> Connecting to Elasticsearch:", end=" ")
            logger.mesg(f"[{ELASTIC_ENVS['host']}]")

        self.client = Elasticsearch(
            hosts=ELASTIC_ENVS["host"],
            ca_certs=ELASTIC_ENVS["ca_certs"],
            api_key=ELASTIC_ENVS["api_key"],
            # basic_auth=(ELASTIC_ENVS["username"], ELASTIC_ENVS["password"]),
        )
        
        if self.verbose:
            logger.success(f"+ Connected:")
            logger.mesg(self.client.info())


if __name__ == "__main__":
    es = ElasticSearchClient()
    es.connect()

    # python -m elastics.client
