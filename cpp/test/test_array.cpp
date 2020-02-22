#include <iostream>

#include "catch.hpp"

#include "Array.hpp"


TEMPLATE_TEST_CASE (
  "ensure constructors and copy operators work",
  "[Array][cuda][unit][copy][constructor]",
  float, double, char
)
{
  SECTION ("Empty Constructor") {
    Array<TestType> arr;
    REQUIRE(arr.data == NULL);
    REQUIRE(arr.IsRef == 0);
    REQUIRE(arr.rank == 0);
  }

  SECTION ("Copy Constructor") {
    Array<TestType> arr;
    Array<TestType> arr1(arr);
    REQUIRE(arr1.IsRef == 1);
  }

  SECTION ("Copy Operator") {
    Array<TestType> arr;
    Array<TestType> arr1 = arr;
    REQUIRE(arr1.IsRef == 1);
  }
}

TEMPLATE_TEST_CASE (
  "Ensure resize works",
  "[Array][cuda][unit][resize]",
  float, double, char
)
{
  SECTION ("resize works on host array") {
    Array<TestType> arrh;
    arrh.resize(10);
  }
}
