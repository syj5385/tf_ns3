/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * Copyright (c) 2014 Natale Patriciello <natale.patriciello@gmail.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#ifndef TCPBIC_H
#define TCPBIC_H

#include "ns3/tcp-congestion-ops.h"
#include "ns3/nstime.h"
#include "ns3/random-variable-stream.h"
#include "ns3/traced-value.h"


#include "dqn_model.h"

class TcpBicIncrementTest;
class TcpBicDecrementTest;

namespace ns3 {

class TcpBic : public TcpCongestionOps
{
public:
  /**
   * \brief Get the type ID.
   * \return the object TypeId
   */
  static TypeId GetTypeId (void);

  /**
   * Copy constructor.
   * \param sock The socket to copy from.
   */
  TcpBic ();
  TcpBic (const TcpBic &sock);

  virtual std::string GetName () const;
  virtual void PktsAcked (Ptr<TcpSocketState> tcb, uint32_t segmentsAcked, const Time& rtt);
  virtual void IncreaseWindow (Ptr<TcpSocketState> tcb,
                               uint32_t segmentsAcked);
  virtual uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                                uint32_t bytesInFlight);

  virtual Ptr<TcpCongestionOps> Fork ();

protected:

  virtual uint32_t Update (Ptr<TcpSocketState> tcb);

private:
  DQN_model model;
  vector<TrainingSet*> t_set;
  Time lastTrainTime;
  uint32_t step = 0;
  float Uprev;
  TracedValue<SequenceNumber32> maxTxtmp;
  TracedValue<SequenceNumber32> lastTx;



  void CCA_train(Ptr<TcpSocketState> tcb, const Time& rtt, DQN_model model);

};

} // namespace ns3
#endif // TCPBIC_H
