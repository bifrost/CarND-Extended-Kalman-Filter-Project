#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF()
{
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
      0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
      0, 0.0009, 0,
      0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  H_laser_ << 1, 0, 0, 0,
      0, 1, 0, 0;

  //create a 4D state vector, we don't know yet the values of the x state
  ekf_.x_ = VectorXd(4);
  ekf_.x_ << 1, 1, 1, 1;

  //state covariance matrix P
  ekf_.P_ = MatrixXd(4, 4);

  //the initial transition matrix F_
  ekf_.F_ = MatrixXd(4, 4);
  ekf_.F_ << 1, 0, 1, 0,
      0, 1, 0, 1,
      0, 0, 1, 0,
      0, 0, 0, 1;

  //measurement matrix
  ekf_.H_ = MatrixXd(2, 4);

  //measurement covariance
  ekf_.R_ = MatrixXd(2, 2);

  //the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);

  //set the acceleration noise components
  noise_ax = 30;
  noise_ay = 20;

  Tools tools;
}

// Experiment radar and laser measurements, for noise_ax=9, noise_ay=9
// RMSE
// x       y       vx      vy       sum (x+y)
// 0.0973, 0.0855, 0.4513, 0.4399 - 0.1828

// Experiment disable radar measurements, for noise_ax=9, noise_ay=9
// RMSE
// x       y       vx      vy       sum (x+y)
// 0.1838, 0.1542, 0.6051, 0.4858 - 0.3380

// Experiment disable laser measurements, for noise_ax=9, noise_ay=9
// RMSE
// x       y       vx      vy       sum (x+y)
// 0.2248, 0.3357, 0.6006, 0.7483 - 0.5605

// Conclusion:
// The RMSE has improved at least a factor two by using both laser and radar measurements.
// One reason could be that we have twice as much data to calculate the position.
// The result might be improved if we were able to collect twice as much laser data instead of radar data.


// Experiments for different noise_ax, noise_ay values, * indicate optimal parameter pairs.
//
// RMSE             noise
// x       y        ax,ay   sum (x+y)
// 0.0973, 0.0855 -  9, 9 - 0.1828

// 0.0942, 0.0836 - 12,12 - 0.1778
// 0.0929, 0.0832 - 14,14 - 0.1761
// 0.0912, 0.0832 - 18,18 - 0.1744
// 0.0904, 0.0830 - 28,20 - 0.1734 *
// 0.0909, 0.0825 - 30,18 - 0.1734 *
// 0.0907, 0.0827 - 30,19 - 0.1734 *
// 0.0905, 0.0829 - 30,20 - 0.1734 *
// 0.0907, 0.0827 - 31,19 - 0.1734 *
// 0.0910, 0.0825 - 31,18 - 0.1735
// 0.0905, 0.0830 - 29,20 - 0.1735
// 0.0906, 0.0830 - 31,20 - 0.1736
// 0.0909, 0.0826 - 28,18 - 0.1735
// 0.0915, 0.0822 - 30,16 - 0.1737
// 0.0902, 0.0836 - 26,22 - 0.1738
// 0.0909, 0.0829 - 22,18 - 0.1738
// 0.0909, 0.0828 - 23,18 - 0.1737
// 0.0910, 0.0829 - 21,18 - 0.1739
// 0.0899, 0.0841 - 24,24 - 0.1740
// 0.0901, 0.0836 - 24,22 - 0.1737
// 0.0905, 0.0831 - 24,20 - 0.1736
// 0.0908, 0.0830 - 24,19 - 0.1738
// 0.0909, 0.0827 - 24,18 - 0.1736
// 0.0914, 0.0824 - 24,16 - 0.1738
// 0.0905, 0.0832 - 25,20 - 0.1737
// 0.0905, 0.0832 - 23,20 - 0.1737

// Conclusion:
// The noise_ax, noise_ay was not set to optimal values and RMSE has been improved by
// 0.0068 for x and 0.0029 for y, by setting noise_ax = 30 and noise_ay = 20.


/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack)
{
  //compute the time elapsed between the current and previous measurements
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  previous_timestamp_ = measurement_pack.timestamp_;

  // do not trust the process if dt > 3 seconds, even if dt is part of the process covariance matrix Q
  // this is needed to switch between the two datasets on the fly
  if (abs(dt) > 3)
  {
    is_initialized_ = false;
  }
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_)
  {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      float ro = measurement_pack.raw_measurements_[0];
      float theta = measurement_pack.raw_measurements_[1];
      float ro_dot = measurement_pack.raw_measurements_[2];

      float x = sin(theta) * ro;
      float y = cos(theta) * ro;
      float x_dot = sin(theta) * ro_dot;
      float y_dot = cos(theta) * ro_dot;

      ekf_.x_ << x, y, x_dot, y_dot;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
    {
      /**
      Initialize state.
      */
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    //init state covariance matrix P
    ekf_.P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
   TODO:
     * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
     * Update the process noise covariance matrix.
     * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float dt_2 = dt * dt;
  float dt_3 = dt_2 * dt;
  float dt_4 = dt_3 * dt;

  //Modify the F matrix so that the time is integrated
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  //set the process covariance matrix Q
  ekf_.Q_ = MatrixXd(4, 4);
  ekf_.Q_ << dt_4 / 4 * noise_ax, 0, dt_3 / 2 * noise_ax, 0,
      0, dt_4 / 4 * noise_ay, 0, dt_3 / 2 * noise_ay,
      dt_3 / 2 * noise_ax, 0, dt_2 * noise_ax, 0,
      0, dt_3 / 2 * noise_ay, 0, dt_2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;

    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  }
  else
  {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;

    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
