"""
Database Utilities for Hiring Pipeline Predictor
Uses the gimme package for MongoDB access following HiredScore patterns.
"""

import logging
from typing import Optional, List, Dict, Any
from pymongo.database import Database
from pymongo.cursor import Cursor

# HiredScore gimme imports
from hs_gimme.db_facade.db_facade_factory import (
    get_mongo_client_db,
    get_raw_mongo_client_db,
)
from hs_gimme.logging_service.logging_service import LoggingService

# HiredScore constants - using proper gimme package classes
from hs_gimme.constants.objects.application import Application
from hs_gimme.constants.objects.req import Req

logger = LoggingService(logging.getLogger(__name__))


class HiringDataClient:
    """
    Client for accessing hiring data from MongoDB.
    Provides methods to fetch applications, requisitions, and related data.
    Uses constants from hs_gimme for field names.
    """

    # Collection names
    APPLICATION_COLLECTION = "application"
    REQ_COLLECTION = "req"

    def __init__(self, environment: str, account_id: str):
        """
        Initialize the data client with environment and account.

        Args:
            environment: The deployment environment (e.g., 'production', 'staging')
            account_id: The account/client identifier
        """
        self.environment = environment
        self.account_id = account_id
        self._db: Optional[Database] = None
        self._raw_db: Optional[Database] = None

    @property
    def db(self) -> Database:
        """Get the applicative MongoDB database connection."""
        if self._db is None:
            self._db = get_mongo_client_db(
                environment=self.environment,
                account_id=self.account_id,
                connect=True
            )
        return self._db

    @property
    def raw_db(self) -> Database:
        """Get the raw MongoDB database connection."""
        if self._raw_db is None:
            self._raw_db = get_raw_mongo_client_db(
                environment=self.environment,
                account_id=self.account_id,
                connect=True
            )
        return self._raw_db

    def get_applications_cursor(
        self,
        query: Optional[Dict] = None,
        projection: Optional[Dict] = None,
        batch_size: int = 1000
    ) -> Cursor:
        """
        Get a cursor for iterating over applications.

        Args:
            query: MongoDB query filter
            projection: Fields to include/exclude
            batch_size: Number of documents per batch

        Returns:
            MongoDB cursor for applications
        """
        query = query or {}
        return self.db[self.APPLICATION_COLLECTION].find(
            query, projection
        ).batch_size(batch_size)

    def get_requisitions_cursor(
        self,
        query: Optional[Dict] = None,
        projection: Optional[Dict] = None,
        batch_size: int = 500
    ) -> Cursor:
        """
        Get a cursor for iterating over requisitions.

        Args:
            query: MongoDB query filter
            projection: Fields to include/exclude
            batch_size: Number of documents per batch

        Returns:
            MongoDB cursor for requisitions
        """
        query = query or {}
        return self.db[self.REQ_COLLECTION].find(
            query, projection
        ).batch_size(batch_size)

    def get_closed_successful_requisitions(
        self,
        additional_filters: Optional[Dict] = None
    ) -> Cursor:
        """
        Get all closed requisitions that resulted in a successful hire.
        Uses Req constants from hs_gimme.

        Args:
            additional_filters: Additional query filters to apply

        Returns:
            Cursor of successful requisitions
        """
        # Requisitions with status indicating successful close
        query = {
            "$or": [
                {Req.REQ_STATUS: {"$in": ["Closed", "Filled", "Closed - Filled"]}},
                {Req.FILL_DATE: {"$exists": True, "$ne": None}},
            ]
        }

        if additional_filters:
            query.update(additional_filters)

        projection = {
            "_id": 1,
            Req.REQ_STATUS: 1,
            Req.FILL_DATE: 1,
            Req.CLOSE_DATE: 1,
            Req.OPEN_DATE: 1,
            Req.JOB_TITLE: 1,
            Req.TOP_CATEGORY: 1,
            Req.EXTERNAL_JOB_BAND: 1,
            Req.COUNTRY: 1,
        }

        return self.get_requisitions_cursor(query, projection)

    def count_applications_for_req(self, req_id: str) -> int:
        """
        Count unique candidates who applied to a requisition.
        Uses Application.REQ_ID constant.

        Args:
            req_id: The requisition ID

        Returns:
            Count of unique candidates
        """
        return self.db[self.APPLICATION_COLLECTION].count_documents(
            {Application.REQ_ID: req_id}
        )

    def get_applications_for_req(
        self,
        req_id: str,
        projection: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Get all applications for a specific requisition.
        Uses Application.REQ_ID constant.

        Args:
            req_id: The requisition ID
            projection: Fields to include/exclude

        Returns:
            List of application documents
        """
        query = {Application.REQ_ID: req_id}
        return list(self.db[self.APPLICATION_COLLECTION].find(query, projection))

    def get_unique_candidate_count_for_req(self, req_id: str) -> int:
        """
        Get count of unique candidates for a requisition.
        Uses Application constants.

        Args:
            req_id: The requisition ID

        Returns:
            Number of unique candidates
        """
        pipeline = [
            {"$match": {Application.REQ_ID: req_id}},
            {"$group": {"_id": f"${Application.CANDIDATE_ID}"}},
            {"$count": "total"}
        ]
        result = list(self.db[self.APPLICATION_COLLECTION].aggregate(pipeline))
        return result[0]["total"] if result else 0

    def get_application_by_id(self, application_id: str) -> Optional[Dict]:
        """
        Get a single application by ID.

        Args:
            application_id: The application ID

        Returns:
            Application document or None
        """
        return self.db[self.APPLICATION_COLLECTION].find_one({"_id": application_id})

    def get_requisition_by_id(self, req_id: str) -> Optional[Dict]:
        """
        Get a single requisition by ID.

        Args:
            req_id: The requisition ID

        Returns:
            Requisition document or None
        """
        return self.db[self.REQ_COLLECTION].find_one({"_id": req_id})

    def aggregate_applications(
        self,
        pipeline: List[Dict],
        allow_disk_use: bool = True
    ) -> Cursor:
        """
        Run an aggregation pipeline on applications.

        Args:
            pipeline: MongoDB aggregation pipeline
            allow_disk_use: Allow disk use for large aggregations

        Returns:
            Aggregation cursor
        """
        return self.db[self.APPLICATION_COLLECTION].aggregate(
            pipeline, allowDiskUse=allow_disk_use
        )

    def aggregate_requisitions(
        self,
        pipeline: List[Dict],
        allow_disk_use: bool = True
    ) -> Cursor:
        """
        Run an aggregation pipeline on requisitions.

        Args:
            pipeline: MongoDB aggregation pipeline
            allow_disk_use: Allow disk use for large aggregations

        Returns:
            Aggregation cursor
        """
        return self.db[self.REQ_COLLECTION].aggregate(
            pipeline, allowDiskUse=allow_disk_use
        )


def create_data_client(environment: str, account_id: str) -> HiringDataClient:
    """
    Factory function to create a HiringDataClient.

    Args:
        environment: The deployment environment
        account_id: The account/client identifier

    Returns:
        Configured HiringDataClient instance
    """
    logger.info(
        "Creating HiringDataClient",
        environment=environment,
        account_id=account_id
    )
    return HiringDataClient(environment, account_id)
