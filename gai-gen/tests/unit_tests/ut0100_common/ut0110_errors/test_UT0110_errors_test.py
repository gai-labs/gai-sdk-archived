import unittest
from fastapi import HTTPException
from gai.common.errors import ApiException
import requests  # replace with the correct import path
import httpx
from unittest.mock import AsyncMock

class UT0110ErrorsTest(unittest.IsolatedAsyncioTestCase):


    def setUp(self):
        # This method will be run before each test
        self.mock_client = AsyncMock()
        self.mock_client.get = AsyncMock()

    async def test_UT0111_ApiException(self):
        # Arrange
        request = httpx.Request(method="GET", url='http://localhost:12031/ApiException')
        self.mock_client.get.return_value = httpx.Response(
            status_code=500,
            json={"detail": "An error occurred."},
            request=request
        )
        # Act
        response = await self.mock_client.get('http://localhost:12031/ApiException')
        # Assert
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "An error occurred."})
        self.assertEqual(response.url, 'http://localhost:12031/ApiException')

    async def test_UT0112_InternalException(self):
        # Arrange
        request = httpx.Request(method="GET", url='http://localhost:12031/InternalException')
        self.mock_client.get.return_value = httpx.Response(
            status_code=500,
            json={"detail": "An internal error occurred."},
            request=request
        )
        # Act
        response = await self.mock_client.get('http://localhost:12031/InternalException')
        # Assert
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "An internal error occurred."})
        self.assertEqual(str(response.url), 'http://localhost:12031/InternalException')

    async def test_UT0113_JSONResponse(self):
        # Arrange
        request = httpx.Request(method="GET", url='http://localhost:12031/JSONResponse')
        self.mock_client.get.return_value = httpx.Response(
            status_code=200,
            json={"message": "Success"},
            request=request
        )
        # Act
        response = await self.mock_client.get('http://localhost:12031/JSONResponse')
        # Assert
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Success"})
        self.assertEqual(str(response.url), 'http://localhost:12031/JSONResponse')

    async def test_UT0114_MessageNotFoundException(self):
        # Arrange
        request = httpx.Request(method="GET", url='http://localhost:12031/MessageNotFoundException/abcd')
        self.mock_client.get.return_value = httpx.Response(
            status_code=404,
            json={"detail": "Message not found"},
            request=request
        )
        # Act
        response = await self.mock_client.get('http://localhost:12031/MessageNotFoundException/abcd')
        # Assert
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Message not found"})
        self.assertEqual(str(response.url), 'http://localhost:12031/MessageNotFoundException/abcd')

    async def test_UT0115_UserNotFoundException(self):
        # Arrange
        request = httpx.Request(method="GET", url='http://localhost:12031/UserNotFoundException/1234')
        self.mock_client.get.return_value = httpx.Response(
            status_code=404,
            json={"detail": "User not found"},
            request=request
        )
        # Act
        response = await self.mock_client.get('http://localhost:12031/UserNotFoundException/1234')
        # Assert
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "User not found"})
        self.assertEqual(str(response.url), 'http://localhost:12031/UserNotFoundException/1234')

    async def test_UT0116_GeneratorMismatchException(self):
        # Arrange
        request = httpx.Request(method="GET", url='http://localhost:12031/GeneratorMismatchException')
        self.mock_client.get.return_value = httpx.Response(
            status_code=400,
            json={"detail": "Generator mismatch"},
            request=request
        )
        # Act
        response = await self.mock_client.get('http://localhost:12031/GeneratorMismatchException')
        # Assert
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Generator mismatch"})
        self.assertEqual(str(response.url), 'http://localhost:12031/GeneratorMismatchException')

    async def test_UT0117_ContextLengthExceededException(self):
        # Arrange
        request = httpx.Request(method="GET", url='http://localhost:12031/ContextLengthExceededException')
        self.mock_client.get.return_value = httpx.Response(
            status_code=413,
            json={"detail": "Context length exceeded"},
            request=request
        )
        # Act
        response = await self.mock_client.get('http://localhost:12031/ContextLengthExceededException')
        # Assert
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json(), {"detail": "Context length exceeded"})
        self.assertEqual(str(response.url), 'http://localhost:12031/ContextLengthExceededException')


if __name__ == '__main__':
    unittest.main()
