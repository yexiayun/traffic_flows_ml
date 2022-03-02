// Copyright (c) 2022, Bongmi
// All rights reserved
// Author: yexiayun@bongmi.com

import UIKit
import Foundation
import RxSwift
import RxCocoa
import SwiftyFoundation
import BMUserBusinessLib
import SnapKit
import UIComponent
import BMBaseWidgetLib
import BMBaseToolsLib
import MJRefresh
import BMCommonWidgetLib
import SensorsAnalyticsSDK
import PassKit
import BMTFemometerUIKit

class CashRegisterViewController: BaseViewController {
  
  enum DialogType {
    case addingExpress
    case addingPayMethod
    case redeemPoint
    case rewardQuestion
    case none
  }
  
  public override var navBarStyle: NavigationBarStyle {
    return .normal
  }
  
  private lazy var scrollView = UIScrollView.instance {
    $0.showsVerticalScrollIndicator = false
    let bottom = (20 + BMBTCommonUtil.bmbt_safeAreaInset().bottom + 45)
    $0.contentInset = UIEdgeInsets(top: 0, left: 0, bottom: bottom, right: 0)
  }
  
  private lazy var shoppingBgView = UIView.instance {
    $0.layer.cornerRadius = 9
    $0.layer.masksToBounds = true
    $0.backgroundColor = ColorGuide.tintBackground
  }
  
  private lazy var submitButton = UIButton.instance {
    $0.backgroundColor = ColorGuide.mainBackground
    $0.setLocalizedTitle(LocalizedText("Pay now"), for: .normal)
    $0.setBackgroundColor(ColorGuide.main, for: .normal)
    $0.setTitleColor(.white, for: .normal)
    $0.layer.cornerRadius = 45 / 2.0
    $0.layer.masksToBounds = true
  }
    
  private lazy var mainBgView = UIView.instance {
    $0.layer.cornerRadius = 9
    $0.layer.masksToBounds = true
    $0.backgroundColor = ColorGuide.tintBackground
  }
  
  private lazy var expressEmptyItemView = CashRegisterItemView.instance {
    $0.title = LocalizedText("Code.ContentShopAddress").value
    $0.backgroundColor = ColorGuide.tintBackground
    $0.cpCornerRadius = 10
    $0.isBottomLineHidden = true
  }
  
  private lazy var expressView: MallAddressDisplayView = {
    let view = MallAddressDisplayView(model: nil, needWave: true)
    view.colorWhenHasAddress = ColorGuide.tintBackground
    view.heightWhenHasAddress = 115.5
    return view
  }()
  
  private lazy var shoppingTipsLabel = UILabel.instance {
    $0.font = FontGuide.poppinsMedium16
    $0.textColor = ColorGuide.normalTitle
    $0.localizedText = LocalizedText("Code.MeTextShopping")
  }
  
  private lazy var itemsCountLabel = UILabel.instance {
    $0.font = FontGuide.regular14
    $0.textColor = ColorGuide.secondTitle
    $0.text = "-"
  }
  
  private lazy var productsCollectionView: UICollectionView = {
    let layout = UICollectionViewFlowLayout()
    layout.scrollDirection = .horizontal
    layout.minimumInteritemSpacing = 7.5
    let view = UICollectionView(frame: .zero,
                                collectionViewLayout: layout)
    view.backgroundColor = ColorGuide.tintBackground
    view.delegate = self
    view.dataSource = self
    view.register(CashRegisterProductCell.self,
                  forCellWithReuseIdentifier: CashRegisterProductCell.bmt_reuseId())
    view.contentInset = UIEdgeInsets(top: 0,
                                     left: 15,
                                     bottom: 0,
                                     right: 0)
    view.showsHorizontalScrollIndicator = false
    return view
  }()
  
  private lazy var payMethodItemView = CashRegisterItemView.instance {
    $0.title = LocalizedText("Code.ShopCheakoutPayment").value
  }
  
  private lazy var couponItemView = CashRegisterItemView.instance {
    $0.title = LocalizedText("Code.ShopCheakoutCoupon").value
  }
  
  private lazy var pointsItemView = CashRegisterItemView.instance {
    $0.title = LocalizedText("Code.Bonus").value
  }
  
  private lazy var subtotalLabel = UILabel.instance {
    $0.font = FontGuide.medium14
    $0.textColor = ColorGuide.normalTitle
    $0.text = LocalizedText("Code.ContentShopSubtotal").value
  }
  
  private lazy var totalPriceLabel = UILabel.instance {
    $0.font = FontGuide.medium19
    $0.textColor = ColorGuide.normalTitle
  }
  
  private lazy var productTotalPriceLabel = UILabel.instance {
    $0.font = FontGuide.regular12
    $0.textColor = ColorGuide.secondTitle
    $0.text = "-"
  }
  
  private lazy var expressFeeLabel = UILabel.instance {
    $0.font = FontGuide.regular12
    $0.textColor = ColorGuide.secondTitle
    $0.text = "-"
  }
  
  private lazy var discountFeeLabel = UILabel.instance {
    $0.font = FontGuide.regular12
    $0.textColor = ColorGuide.secondTitle
    $0.text = "-"
  }
  
  private lazy var rewardTipsLabel = UILabel.instance {
    $0.font = FontGuide.regular12
    $0.text = "-"
    $0.textColor = ColorGuide.normalTitle
  }
  
  private lazy var rewardQuestionButton = UIButton.instance {
    $0.setImage(Bundle.bmCommon_IMG("common/btn_info_white_s"),
                for: .normal)
    $0.frame = CGRect(x: 0,
                      y: 0,
                      width: 44,
                      height: 44)
    $0.setTouchAreaEdgeInset(UIEdgeInsets(top: 10,
                                          left: 10,
                                          bottom: 10,
                                          right: 10))
  }
  
  private lazy var expressBgView = UIView()
  
  private lazy var orderManager = MallOrderManager(account: account)
  private lazy var couponsManger = MallCouponsManager(account: account)
  private lazy var pointsRedeemManager = MallPointRedeemManager(account: account)
  
  private var submitCallback: ((PaymentResult) -> Void)?
  private var expressSelected = BehaviorRelay<BMTExpressAddress?>(value: nil)
  private var couponsSorted: [CouponDisplayItem] = []
  private var couponSelected = BehaviorRelay<CouponDisplayItem?>(value: nil)
  private var pointItemCanUse: PointRedeemDisplayItem?
  private var pointItemSelected = BehaviorRelay<PointRedeemDisplayItem?>(value: nil)
  private var dialogType: DialogType = .none
  private var account: BMUBFemometerEntityAccount!
  private var cartItems: [CartProductItem]!
  
  public convenience init(items: [CartProductItem]) {
    self.init()
    cartItems = items
  }
  
  override func viewDidLoad() {
    super.viewDidLoad()
    guard let account = AppManager.shared.account else {
      EasyAlert.alert(message: "Account Not Found")
      return
    }
    self.account = account
    orderManager = MallOrderManager(account: account)
    couponsManger = MallCouponsManager(account: account)
    pointsRedeemManager = MallPointRedeemManager(account: account)
    orderManager.cartItems = cartItems
    setUpUI()
    setUpEvents()
    downloadBaseData()
    setUpDefaultData()
    setUpRx()
  }
  
  private func downloadBaseData() {
    // MARK: - 初始数据
    
    view.hud.showLoading()
    let group = DispatchGroup()
    
    // 下载快递收件信息
    group.enter()
    orderManager
      .requestAddressListTask()
      .subscribeNext { [weak self] result in
        switch result {
        case .success(let expressModels):
          if let defaultExpress = expressModels.first?.payload {
            self?.expressSelected.accept(defaultExpress)
          }
        case .failure(let error):
          self?.view.hud.show(text: error.localizedDescription)
        }
        group.leave()
      }
      .disposed(by: rxDisposeBag)
    
    // 下载优惠券信息
    group.enter()
    couponsManger
      .downloadCouponsThenSort(cartItems: orderManager.cartItems) { [weak self] couponItems in
        self?.couponsSorted = couponItems
        if let maxDeductItem = couponItems.first {
          // 业务要求默认选择优惠最大的那张
          self?.couponSelected.accept(maxDeductItem)
        }
        group.leave()
      }
    
    // 下载可用积分
    group.enter()
    pointsRedeemManager.refreshPoints { [weak self] (points, error) in
      if let error = error {
        self?.view.hud.show(text: error.localizedDescription)
      } else {
        self?.caculatePointsCanUse()
      }
      group.leave()
    }
    
    // 是否有历史订单
    group.enter()
    orderManager.checkHasHistoryOrder { hasHistory, error in
      group.leave()
    }

    group.notify(queue: DispatchQueue.main) {
      self.view.hud.hideLoading()
    }
  }
  
  private func setUpDefaultData() {
    // item number
    let count = orderManager.cartItems.count
    var tail = LocalizedText("Code.ShopOrdersItems").value
    if count == 1 {
      tail = ""
    }
    itemsCountLabel.text = "\(count) \(tail)"
    // 商品总价
    let productPricePrefix = LocalizedText("Code.ShopPaylistTotal").value
    let totalProductPrice = orderManager.caculalteOriginTotalPrice()
    productTotalPriceLabel.text = String(format: "\(productPricePrefix)%.2f",
                                         totalProductPrice.doubleValue / 100.0)
    
    // 运费
    let freightPrefix = LocalizedText("Code.ShopPaylistFreight").value
    let freight = orderManager.caculateFinalExpressFee()
    expressFeeLabel.text = String(format: "\(freightPrefix)%.2f",
                                  freight.doubleValue / 100.0)
    
    // reward info
    if account.isPrime() {
      rewardTipsLabel.localizedText = LocalizedFormatText("Code.ShopCheakoutRewardPoints", 1000)
    } else {
      rewardTipsLabel.localizedText = LocalizedFormatText("Code.ShopCheakoutRewardPoints", 500)
    }
  }
  
  private func setUpEvents() {
    expressView.tapped = { [weak self] in
      guard let self = self else { return }
      let vc = MallAddressListViewController(account: self.account)
      vc.selectAddressCallback = { [weak self] address in
        self?.expressSelected.accept(address)
      }
      self.pushVC(vc)
    }
    
    expressEmptyItemView.tapped = { [weak self] in
      guard let self = self else { return }
      self.gotoAddingNewAddress()
      // 新增
    }
    
    payMethodItemView.tapped = { [weak self] in
      
    }
    
    couponItemView.tapped = { [weak self] in
      
    }
    
    rewardQuestionButton.rx
      .tap
      .subscribeNext { [weak self] in
        self?.showRewardQuestionTipsDialog()
      }
      .disposed(by: rxDisposeBag)
    
    pointsItemView.tapped = { [weak self] in
      self?.showRedeemPointsDialog()
    }
    
    submitButton.rx
      .tap
      .subscribeNext { [weak self] in
        guard let self = self else { return }
        guard self.expressSelected.value.isSome else {
          return
        }
        
        self.orderManager.submit { [weak self] result in
          guard let self = self else { return }
          switch result {
          case .success(let orderInfo):
            self.gotoOrder()
          case .failure(let error):
            self.view.hud.show(text: error.localizedDescription)
            self.gotoOrder()
          }
          self.updateHasHistoryOrderStatus()
        }
      }
      .disposed(by: rxDisposeBag)
  }
  
  private func setUpRx() {
    
    expressSelected
      .subscribeNext { [weak self] express in
        guard let self = self else { return }
        self.expressView.expressModel = express
        self.orderManager.express = express
        self.refresExpressUI()
      }
      .disposed(by: rxDisposeBag)
    
    couponSelected
      .subscribeNext { [weak self] coupon in
        guard let self = self else { return }
        self.orderManager.coupon = coupon
        self.couponItemView.subTitle = coupon?.entity.name
        self.couponItemView.rightTitle = coupon?.realSubPriceString
        // 优惠券不同，积分会被影响
        self.caculatePointsCanUse()
      }
      .disposed(by: rxDisposeBag)
    
    pointItemSelected
      .filter { $0.isSome }
      .subscribeNext { [weak self] (pointsItem) in
        guard let self = self else { return }
        guard let pointsItem = pointsItem, pointsItem.reductPoints > 0 else {
          return
        }
        self.orderManager.pointRedeemItem = pointsItem
        let tail = LocalizedText("Code.ShopCheakoutPoints").value
        self.pointsItemView.subTitle = "\(pointsItem.reductPoints)\(tail)"
        self.pointsItemView.rightTitle = "-\(pointsItem.amount.doubleValue / 100.0)"
      }
      .disposed(by: rxDisposeBag)
    
    // 优惠券/ 积分不同选择，引起最终总价的改变
    Observable.combineLatest(couponSelected, pointItemSelected)
      .subscribeNext { [weak self] (coupon, points) in
        self?.caculateTotalPriceThenRefreshUI()
        self?.caculateTotalReductThenRefreshUI()
      }
      .disposed(by: rxDisposeBag)
  }
  
  private func showRedeemPointsDialog() {
    guard let pointItemCanUse = pointItemCanUse else {
      return
    }
    
    guard pointItemSelected.value.isNone else {
      return
    }
    dialogType = .redeemPoint
    showDialog(with: pointItemCanUse.text,
               actionTitle: LocalizedText("Code.BonusTextRedeem").value)
  }
  
  private func showAddingNewExpressDialog() {
    
    guard expressSelected.value.isNone else {
      return
    }
    dialogType = .addingExpress
    showDialog(with: LocalizedText("Code.ShopCheakoutNoAddress").value,
               actionTitle: LocalizedText("Code.BonusTextRedeem").value)
  }
  
  private func showAddingPayMethodDialog() {
    
    dialogType = .addingPayMethod
    showDialog(with: LocalizedText("Code.ShopCheakoutNoPayment").value,
               actionTitle: LocalizedText("Code.ShopCheakoutAddPayment").value)
  }
  
  private func showRewardQuestionTipsDialog() {
    dialogType = .rewardQuestion
    showDialog(with: LocalizedText("Code.ShopCheakoutRewardInformation").value,
               actionTitle: nil)
  }
  
  private func showDialog(with content: String,
                          actionTitle: String?) {
    let model = BMBWDialogViewModel()
    model.content = content
    if let actionTitle = actionTitle {
      model.actionBtnTitle = actionTitle
    }
    let dialogView = BMBWDialogView(style: .onlyTitle,
                              dialogModel: model)
    dialogView.contentLabel.textAlignment = .center
    dialogView.delegate = self
    dialogView.show(on: view)
  }
  
  private func caculateTotalPriceThenRefreshUI() {
    let totalPrice = orderManager.caculateFinalTotalPrice()
    let safeCurrency = orderManager.cartItems.first?.product.currency ?? ""
    let priceAttr = NSMutableAttributedString(string: "\(safeCurrency)\(totalPrice.doubleValue / 100.0)")
    priceAttr.addAttributes([NSMutableAttributedString.Key.font: FontGuide.medium12],
                            range: NSRange(location: 0, length: safeCurrency.count))
    totalPriceLabel.attributedText = priceAttr
  }
  
  private func caculateTotalReductThenRefreshUI() {
    let deductPrefix = LocalizedText("Code.ShopPaylistDiscount").value
    let deduct = orderManager.caculateTotalDeduct()
    discountFeeLabel.text = String(format: "\(deductPrefix)%.2f",
                                   deduct.doubleValue / 100.0)
  }
  
  private func caculatePointsCanUse() {
    let totalPrice = orderManager.caculateTotalPirceUsingCouponAndFreight()
    let item = pointsRedeemManager.getPointDisplayItemsCanUse(in: orderManager.cartItems,
                                                   finalTotalPrice: totalPrice)
    pointItemCanUse = item
    if pointItemSelected.value.isSome {
      pointItemSelected.accept(pointItemCanUse)
    }
  }
  
  private func updateHasHistoryOrderStatus() {
    view.hud.showLoading()
    orderManager.checkHasHistoryOrder { [weak self] (hasHistoryOrder, error) in
      guard let self = self else { return }
      self.view.hud.hideLoading()
    }
  }
  
  private func gotoAddingNewAddress() {
    let vc = MallEditAddressViewController(account: self.account,
                                           vm: nil)
    vc.finshFlow = { [weak self] (address) in
      self?.expressSelected.accept(address)
    }
    self.pushVC(vc)
  }
  
  private func gotoOrder() {
    guard let account = AppManager.shared.account else {
      return
    }
    if var vcs = navigationController?.viewControllers {
      vcs.removeLast()
      let vc = MyOrdersViewController(account: account)
      vc.hidesBottomBarWhenPushed = true
      vcs.append(vc)
      navigationController?.setViewControllers(vcs, animated: true)
    }
  }
  
  private func refresExpressUI() {
    let hasAddress = expressSelected.value.isSome
    if hasAddress {
      toHasAddressUIStyle()
    } else {
      toNoAddressUIStyle()
    }
  }
  
  private func toHasAddressUIStyle() {
    expressView.isHidden = false
    expressEmptyItemView.isHidden = true
    expressBgView.snp.updateConstraints { make in
      make.height.equalTo(115.5)
    }
  }
  
  private func toNoAddressUIStyle() {
    expressView.isHidden = true
    expressEmptyItemView.isHidden = false
    expressBgView.snp.updateConstraints { make in
      make.height.equalTo(60)
    }
  }
}

/// 布局
extension CashRegisterViewController {
  
  private func setUpUI() {
    view.backgroundColor = ColorGuide.normalBackground
    navigationItem.title = LocalizedText("Code.ShopCart" ).value
    view.addSubview(scrollView)
    scrollView.addSubview(expressBgView)
    expressBgView.addSubview(expressView)
    expressBgView.addSubview(expressEmptyItemView)
    scrollView.addSubview(mainBgView)
    mainBgView.addSubview(shoppingTipsLabel)
    mainBgView.addSubview(itemsCountLabel)
    mainBgView.addSubview(productsCollectionView)
    mainBgView.addSubview(payMethodItemView)
    mainBgView.addSubview(couponItemView)
    mainBgView.addSubview(pointsItemView)
    mainBgView.addSubview(subtotalLabel)
    mainBgView.addSubview(totalPriceLabel)
    mainBgView.addSubview(productTotalPriceLabel)
    mainBgView.addSubview(expressFeeLabel)
    mainBgView.addSubview(discountFeeLabel)
    mainBgView.addSubview(rewardTipsLabel)
    mainBgView.addSubview(rewardQuestionButton)
    
    view.addSubview(submitButton)
    
    scrollView.snp.makeConstraints { make in
      make.edges.equalToSuperview()
    }
    
    expressBgView.snp.makeConstraints { make in
      make.left.equalTo(16)
      make.right.equalTo(-16)
      make.width.equalTo(FrameGuide.screenWidth - 32)
      make.height.equalTo(60)
      make.top.equalTo(10)
    }
    
    expressEmptyItemView.snp.makeConstraints { make in
      make.left.right.equalToSuperview()
      make.height.equalTo(60)
    }
    
    expressView.snp.makeConstraints { make in
      make.height.equalTo(60)
      make.left.right.equalToSuperview()
    }
    
    mainBgView.snp.makeConstraints { make in
      make.left.equalTo(16)
      make.right.equalTo(-16)
      make.width.equalTo(FrameGuide.screenWidth - 32)
      make.top.equalTo(expressBgView.snp.bottom).offset(10)
      make.bottom.equalToSuperview().offset(-20)
    }
    
    shoppingTipsLabel.snp.makeConstraints { make in
      make.top.equalToSuperview().offset(15)
      make.left.equalToSuperview().offset(15)
    }
    
    itemsCountLabel.snp.makeConstraints { make in
      make.right.equalToSuperview().offset(-15)
      make.centerY.equalTo(shoppingTipsLabel)
    }
    
    productsCollectionView.snp.makeConstraints { make in
      let itemWidth = CashRegisterProductCell.itemWidth
      make.top.equalTo(shoppingTipsLabel.snp.bottom).offset(15)
      make.left.right.equalToSuperview()
      make.height.equalTo(itemWidth)
    }
    
    payMethodItemView.snp.makeConstraints { make in
      make.left.right.equalToSuperview()
      make.top.equalTo(productsCollectionView.snp.bottom)
      make.height.equalTo(71.5)
    }
    
    couponItemView.snp.makeConstraints { make in
      make.left.right.equalToSuperview()
      make.top.equalTo(payMethodItemView.snp.bottom)
      make.height.equalTo(71.5)
    }
    
    pointsItemView.snp.makeConstraints { make in
      make.left.right.equalToSuperview()
      make.top.equalTo(couponItemView.snp.bottom)
      make.height.equalTo(71.5)
    }
    
    subtotalLabel.snp.makeConstraints { make in
      make.left.equalTo(15)
      make.top.equalTo(pointsItemView.snp.bottom).offset(18.5)
    }
    
    totalPriceLabel.snp.makeConstraints { make in
      make.right.equalTo(-15)
      make.centerY.equalTo(subtotalLabel)
    }
    
    productTotalPriceLabel.snp.makeConstraints { make in
      make.right.equalTo(totalPriceLabel)
      make.top.equalTo(totalPriceLabel.snp.bottom).offset(3.5)
    }
    
    expressFeeLabel.snp.makeConstraints { make in
      make.right.equalTo(totalPriceLabel)
      make.top.equalTo(productTotalPriceLabel.snp.bottom).offset(3.5)
    }
    
    discountFeeLabel.snp.makeConstraints { make in
      make.right.equalTo(totalPriceLabel)
      make.top.equalTo(expressFeeLabel.snp.bottom).offset(3.5)
    }
    
    rewardQuestionButton.snp.makeConstraints { make in
      make.width.height.equalTo(15)
      make.right.equalToSuperview().offset(-17)
      make.top.equalTo(discountFeeLabel.snp.bottom).offset(4.5)
      make.bottom.equalTo(-15)
    }
    
    rewardTipsLabel.snp.makeConstraints { make in
      make.right.equalTo(rewardQuestionButton.snp.left).offset(-2.5)
      make.centerY.equalTo(rewardQuestionButton)
    }
    
    submitButton.snp.makeConstraints { make in
      make.height.equalTo(45)
      make.left.equalTo(22.5)
      make.right.equalTo(-22.5)
      make.bottom.equalTo(-(FrameGuide.safeAreaBottomHeight + 27.5))
    }
  }
}

extension CashRegisterViewController: UICollectionViewDelegate,
                                        UICollectionViewDataSource,
                                      UICollectionViewDelegateFlowLayout {
  
  func collectionView(_ collectionView: UICollectionView,
                      numberOfItemsInSection section: Int) -> Int {
    return orderManager.cartItems.count
  }
  
  func collectionView(_ collectionView: UICollectionView,
                      cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
    let reuseId = CashRegisterProductCell.bmt_reuseId()
    let cell = collectionView.dequeueReusableCell(withReuseIdentifier: reuseId!,
                                                  for: indexPath) as! CashRegisterProductCell
    cell.bind(with: orderManager.cartItems[indexPath.row])
    return cell
  }
  
  func collectionView(_ collectionView: UICollectionView,
                      layout collectionViewLayout: UICollectionViewLayout,
                      sizeForItemAt indexPath: IndexPath) -> CGSize {
    let itemWidth = CashRegisterProductCell.itemWidth
    return CGSize(width: itemWidth, height: itemWidth)
  }
}

extension CashRegisterViewController: BMBWDialogViewDelegate  {
  
  public func dialogView(_ dialogView: BMBWDialogView,
                         clickActionBtn sender: UIButton) {
    switch dialogType {
    case .redeemPoint:
      caculatePointsCanUse()
      if let pointItemCanUse = pointItemCanUse {
        pointItemSelected.accept(pointItemCanUse)
      }
    case .addingExpress:
      gotoAddingNewAddress()
    case .addingPayMethod:
      break
    default:
      break
    }
    dialogType = .none
  }
  
  public func dialogView(_ dialogView: BMBWDialogView,
                         clickSecondBtn sender: UIButton) {
  }
  
  public func dialogView(_ dialogView: BMBWDialogView,
                         clickCloseBtn sender: UIButton) {
    dialogType = .none
  }
}
